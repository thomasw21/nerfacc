import argparse
import time
from enum import Enum
from pathlib import Path
from random import choices
from typing import Tuple, List, Callable, Optional

import numpy as np
import torch
import torchvision.transforms
from torchvision.transforms import functional as F
from torch import nn
from torchvision.utils import save_image
from transformers import CLIPModel, CLIPTokenizer, CLIPProcessor, AutoConfig

import nerfacc
from nerfacc import OccupancyGrid, ContractionType

"""
TODOs:
 - Intermediary rendering to check that things are going well
 - Validation at another angle from training. 
"""

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", type=str, required=True, help="Text to fit")
    parser.add_argument("--save-model-path", type=Path, required=True, help="Where to save the model")
    parser.add_argument("--load-model-path", type=Path, help="Where to load the model from")
    parser.add_argument("--save-images-path", type=Path, required=True, help="Where to save the images that we generate at the end")

    ### Training nerf configs
    parser.add_argument(
        "--aabb",
        type=lambda s: [float(item) for item in s.split(",")],
        default="-0.5,-0.5,-0.5,0.5,0.5,0.5",
        help="delimited list input",
    )
    parser.add_argument("--unbounded", action="store_true", help="whether to use unbounded rendering")
    parser.add_argument("--update-occupancy-grid-interval", type=int, default=16, help="Update occupancy grid every n steps")
    parser.add_argument("--training-theta", type=float, default=30, help="Elevation angle you're training at")
    parser.add_argument("--validation-theta", type=float, default=45, help="Elevation angle you're training at")

    ### Optimizer
    # See DreamFields paper
    parser.add_argument("--lambda-transmittance-loss", type=float, default=0.5)
    parser.add_argument("--transmittance-loss-ceil", type=float, default=0.88)
    # Dreamfusion Open source implementation
    parser.add_argument("--lambda-transmittance-entropy", type=float, default=1e-4)
    parser.add_argument("--learning-rate", type=float, default=1e-2, help="Learning rate")
    parser.add_argument("--iterations", type=int, default=2000, help="Number of optimizer steps")
    parser.add_argument("--batch-size", type=int, default=2, help="Number of camera views within a single batch")

    ### Logging
    parser.add_argument("--log-interval", type=int, default=100, help="Log every n steps")

    args = parser.parse_args()
    args.grid_resolution = 128

    args.save_model_path.parent.mkdir(parents=True, exist_ok=True)
    args.save_images_path.parent.mkdir(parents=True, exist_ok=True)

    return args

class TextImageDiscriminator(nn.Module):
    """Clip based"""
    def __init__(self):
        super().__init__()
        model_name = "openai/clip-vit-base-patch32"
        self.model = CLIPModel.from_pretrained(model_name)
        self.tokenizer = CLIPTokenizer.from_pretrained(model_name)
        self.image_processor = CLIPProcessor.from_pretrained(model_name)
        self.config = AutoConfig.from_pretrained(model_name)

    def forward(self, encoded_images: torch.Tensor, encoded_texts: torch.Tensor) -> torch.Tensor:

        ### Copied form `modeling_clip.py`
        # normalized features
        encoded_images = encoded_images / encoded_images.norm(p=2, dim=-1, keepdim=True)
        encoded_texts = encoded_texts / encoded_texts.norm(p=2, dim=-1, keepdim=True)

        # cosine similarity as logits
        # TODO @thomasw21: Check if one needs the actual logit_scale at all.
        # logit_scale = self.model.logit_scale.exp()
        # return - (encoded_texts * encoded_images).sum(-1)

        # # Intuitively you really want to reach one really fast (ie gains from 0.5 to 1 are more important than from 0 to 0.5), we need a more hardcore convex function maybe log(1 - cosine)
        return (encoded_texts * encoded_images).sum(-1)
        # return torch.acos((encoded_texts * encoded_images).sum(-1))

    @property
    def device(self):
        return self.model.text_projection.weight.device

    def encode_texts(self, texts: List[str]) -> torch.Tensor:
        inputs = self.tokenizer(texts, padding=True, return_tensors="pt").to(self.device)
        return self.model.get_text_features(**inputs)

    def encode_images(self, images: torch.Tensor, encoded_texts: torch.Tensor):
        assert len(images) == encoded_texts.shape[0], f"Image: {images.shape}\nEncoded_texts: {encoded_texts.shape}"
        assert images.shape[1] == 3, "RGB images"
        # TODO @thomasw21: This is non differentiable ...
        # inputs = self.image_processor(images=images, return_tensors="pt")
        inputs = {"pixel_values": images}
        return self.model.get_image_features(**inputs)

    @property
    def image_height_width(self) -> Tuple[int, int]:
        return self.config.vision_config.image_size, self.config.vision_config.image_size

class ViewDependentPrompter:
    """Update input text to condition on camera view

    TODO @thomasw21: setup correctly the thing in order to obtain view dependent prompting
    """

    suffixes = ["front view", "side view", "back view", "side view"]
    def __init__(self, text: str):
        self.text = text

    def get_camera_view_prompt(self, phi: float) -> str:
        """Given a homogenous matrix, we return the updated prompt

        TODO @thomasw21: code a generic one using sensor instead of the angles
        """
        quadrants = [30, 180, 210, 360]
        assert len(self.suffixes) == len(quadrants)
        assert quadrants[-1] == 360
        for suffix, angle_limit in zip(self.suffixes, quadrants):
            if phi <= angle_limit:
                return self.get_prompt(suffix)

    def get_prompt(self, suffix: str):
        return f"{self.text}, {suffix}"

    def get_all_text_prompts(self) -> List[str]:
        return [self.get_prompt(suffix) for suffix in self.suffixes]

@torch.no_grad()
def generate_random_360_angles(num_angles: int, device: torch.device, stochastic_angles: bool = True) -> torch.Tensor:
    if stochastic_angles:
        return torch.rand(num_angles, device=device) * 360.0
    else:
        return torch.arange(num_angles, device=device) * (360.0 / num_angles)

Sensors = Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
@torch.no_grad()
def generate_sensors(
    image_height: int,
    image_width: int,
    theta: int,
    phis: torch.Tensor,
) -> Sensors:
    N, = phis.shape
    device = phis.device
    dtype = phis.dtype

    # Angle from equator, 30 degrees from equator
    theta = (90 - theta) * np.pi / 180
    # Rotate according to y-axis
    rot_theta = torch.tensor(
        [
            [np.cos(theta), 0, np.sin(theta)],
            [0, 1, 0],
            [-np.sin(theta), 0, np.cos(theta)],
        ],
        device=device,
        dtype=dtype
    ) # [3, 3]

    sin_angles = torch.sin(phis * np.pi / 180) # [N,]
    cos_angles = torch.cos(phis * np.pi / 180) # [N,]

    # Rotate according to z-axis
    rot_phi = torch.zeros(N, 3, 3, device=device, dtype=dtype)
    rot_phi[:, -1, -1] = 1
    rot_phi[:, 0, 0] = cos_angles
    rot_phi[:, 1, 0] = sin_angles
    rot_phi[:, 0, 1] = -sin_angles
    rot_phi[:, 1, 1] = cos_angles

    rotations = rot_phi @ rot_theta

    # Origins
    radius = 2
    origins = \
        rotations @ torch.tensor([0, 0, radius], device=device, dtype=dtype)
        # + torch.tensor([0.5, 0.5, 0.5], device=angles.device, dtype=angles.dtype) # origin of the bounding box is 0

    # Camera specific values
    camera_angle_x = 45 * np.pi / 180
    ratio = image_width / image_height # make sure that the pixels are actually square
    focal_x = 0.5 * image_height / np.tan(0.5 * camera_angle_x)
    focal_y = focal_x * ratio
    K = torch.tensor(
        [
            [focal_x, 0, image_width * 0.5, 0],
            [0, focal_y, image_height * 0.5, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ],
        device=device,
        dtype=dtype
    )

    return rotations, origins, K

### Rendering helpers

def get_sigma_fn(
    query_density: Callable[[torch.Tensor], torch.Tensor],
    rays_o: torch.Tensor,
    rays_d: torch.Tensor,
) -> Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]:
    def sigma_fn(
        t_starts: torch.Tensor,
        t_ends: torch.Tensor,
        ray_indices: torch.Tensor
    ) -> torch.Tensor:
        """ Query density values from a user-defined radiance field.
        :params t_starts: Start of the sample interval along the ray. (n_samples, 1).
        :params t_ends: End of the sample interval along the ray. (n_samples, 1).
        :params ray_indices: Ray indices that each sample belongs to. (n_samples,).
        :returns The post-activation density values. (n_samples, 1).
        """

        t_origins = rays_o[ray_indices]  # (n_samples, 3)
        t_dirs = rays_d[ray_indices]  # (n_samples, 3)
        # This essentially compute the mean position, we might instead jitter and randomly sample for robustness
        positions = t_origins + t_dirs * (t_starts + t_ends) / 2.0
        sigmas = query_density(positions)
        return sigmas  # (n_samples, 1)
    return sigma_fn

def get_rgb_sigma_fn(
    radiance_field: nn.Module,
    rays_o: torch.Tensor,
    rays_d: torch.Tensor,
) -> Callable[[torch.Tensor, torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
    def rgb_sigma_fn(
        t_starts: torch.Tensor,
        t_ends: torch.Tensor,
        ray_indices: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Query rgb and density values from a user-defined radiance field.
        :params t_starts: Start of the sample interval along the ray. (n_samples, 1).
        :params t_ends: End of the sample interval along the ray. (n_samples, 1).
        :params ray_indices: Ray indices that each sample belongs to. (n_samples,).
        :returns The post-activation rgb and density values.
            (n_samples, 3), (n_samples, 1).
        """
        t_origins = rays_o[ray_indices]  # (n_samples, 3)
        t_dirs = rays_d[ray_indices]  # (n_samples, 3)
        # This essentially compute the mean position, we might instead jitter and randomly sample for robustness
        positions = t_origins + t_dirs * (t_starts + t_ends) / 2.0
        rgbs, sigmas = radiance_field(positions, t_dirs)
        return rgbs, sigmas  # (n_samples, 3), (n_samples, 1)
    return rgb_sigma_fn


def render_images(
    radiance_field: nn.Module,
    query_density: Callable[[torch.Tensor], torch.Tensor],
    occupancy_grid: OccupancyGrid,
    image_height: int,
    image_width: int,
    sensors: Sensors,
) -> Tuple[torch.Tensor, torch.Tensor]:
    device = sensors[0].device
    camera_rotations, camera_centers, camera_intrinsics = sensors
    N = camera_rotations.shape[0]

    # Compute which rays we should run
    # # We for now compute the center of each pixel, thus the 0.5
    # # Get help: https://www.cs.cmu.edu/~16385/s17/Slides/11.1_Camera_matrix.pdf
    assert camera_intrinsics[0,0] == camera_intrinsics[1,1], "focal needs to be the same in x and y"
    x, y = torch.meshgrid(
        torch.arange(image_height, device=device) - camera_intrinsics[0, 2],
        torch.arange(image_width, device=device) - camera_intrinsics[1, 2],
        indexing="ij",
    ) # [H, W]
    pixel_position_in_camera = torch.stack([
        x,
        y,
        - camera_intrinsics[0, 0][None, None].expand(image_height, image_height)
    ], dim=-1) / camera_intrinsics[0,0] # [H, W, 3]
    directions = (camera_rotations[:, None, :, :] @ pixel_position_in_camera.view(1, image_height * image_width, 3, 1)).squeeze(-1) # [N, H, W, 3]
    # pixel_position_in_world = directions + C[:, None, None, :]
    view_dirs = (directions / torch.linalg.norm(directions, dim=-1)[..., None]).view(-1, 3) # [N * image_height * image_width, 3]
    # TODO @thomasw21: Figure out a way without copying data
    origins = torch.repeat_interleave(camera_centers, image_height * image_height, dim=0)

    #### From nerfacc README:
    # Efficient Raymarching: Skip empty and occluded space, pack samples from all rays.
    # packed_info: (n_rays, 2). t_starts: (n_samples, 1). t_ends: (n_samples, 1).
    # TODO @thomasw21: determine the chunk size
    chunk = 2**15
    results = []
    for i in range(0, N * image_height * image_width, chunk):
        origins_shard = origins[i: i + chunk]
        view_dirs_shard = view_dirs[i: i + chunk]
        with torch.no_grad():
            packed_info, t_starts, t_ends = nerfacc.ray_marching(
                rays_o=origins_shard,
                rays_d=view_dirs_shard,
                sigma_fn=get_sigma_fn(query_density, rays_o=origins_shard, rays_d=view_dirs_shard),
                scene_aabb=occupancy_grid.roi_aabb, # TODO @thomasw21: Need to pass it down otherwise this is going to be hell
                grid=occupancy_grid, # This is fucked
                # near_plane=0.2,
                # far_plane=1.0,
                early_stop_eps=1e-4,
                # alpha_thre=1e-2,
                stratified=True
            )

        # Differentiable Volumetric Rendering.
        # colors: (n_rays, 3). opacity: (n_rays, 1). depth: (n_rays, 1).
        color, opacity, depth = nerfacc.rendering(
            rgb_sigma_fn=get_rgb_sigma_fn(radiance_field, rays_o=origins_shard, rays_d=view_dirs_shard),
            packed_info=packed_info,
            t_starts=t_starts,
            t_ends=t_ends,
        )
        results.append((color, opacity, depth))

    color = torch.cat([elt[0] for elt in results], dim=0).view(N, image_height, image_width, 3)
    opacity = torch.cat([elt[1] for elt in results], dim=0).view(N, image_height, image_width, 1)
    return color, opacity

class Background(Enum):
    RANDOM_COLOR_UNIFORM_BACKGROUND = 1
    RANDOM_COLOR_BACKGROUND = 2
    RANDOM_TEXTURE = 3
    CHECKERBOARD = 4
    WHITE = 5
    BLACK = 6

def data_augment(
    color: torch.Tensor,
    opacity: torch.Tensor,
    # TODO @thomasw21: Make random background color accessible through CLI
    background: Optional[Background] = None,
    blur_background: bool = True
) -> Tuple[torch.Tensor, torch.Tensor]:
    N, H, W, _ = color.shape
    # Do random crop
    # TODO @thomasw21: Adding a random crop would really increase data
    img = torch.cat([
        color.permute(3, 0, 1, 2),
        opacity.permute(3, 0, 1, 2)
    ])
    transforms = torchvision.transforms.Compose([
        # DreamFields
        torchvision.transforms.RandomResizedCrop(size=(H, W), scale=(0.80, 1.0))
    ])
    img = transforms(img)
    color = img[:3].permute(1, 2, 3, 0)
    opacity = img[-1:].permute(1, 2, 3, 0)

    # Background
    if background is not None:
        # Single colored background
        if background is background.RANDOM_COLOR_UNIFORM_BACKGROUND:
            background_color = torch.rand(N, 1, 1, 3, device=color.device)
        elif background is background.RANDOM_COLOR_BACKGROUND:
            background_color = torch.rand(N, H, W, 3, device=color.device)
        elif background is background.RANDOM_TEXTURE:
            raise NotImplementedError
        elif background is background.CHECKERBOARD:
            # https://github.com/google-research/google-research/blob/4f54cade26f40728be7fda05c89011f89b7b7b7f/dreamfields/experiments/diffusion_3d/augment.py#L40
            nsq_x = 8
            nsq_y = 8
            assert H % nsq_x == 0
            assert W % nsq_y == 0
            sq_x = H // nsq_x
            sq_y = W // nsq_y
            color1, color2 = torch.rand(2, N, 3, device=color.device)
            background_color = color1[:, None, None, :].repeat(1, H, W, 1).view(N, nsq_x, sq_x, nsq_y, sq_y, 3)
            background_color[:, ::2, :, 1::2, :, :] = color2[:, None, None, None, None, :]
            background_color[:, 1::2, :, ::2, :, :] = color2[:, None, None, None, None, :]
            background_color = background_color.view(N, H, W, 3)
        elif background is background.WHITE:
            background_color = torch.ones(1, 1, 1, 1, device=color.device)
        elif background is background.BLACK:
            background_color = torch.zeros(1, 1, 1, 1, device=color.device)
        else:
            raise ValueError

        if blur_background:
            min_blur, max_blur = (0.0, 10.)
            sigma_x, sigma_y = np.random.rand(2) * (max_blur - min_blur) + min_blur
            background_color = F.gaussian_blur(
                torch.broadcast_to(background_color, (N, H, W, 3)).permute(0, 3, 1, 2),
                kernel_size=[15, 15],
                # Weird, but it's in dreamfields https://github.com/google-research/google-research/blob/00392d6e3bd30bfe706859287035fcd8d53a010b/dreamfields/dreamfields/config/config_base.py#L130
                sigma=[sigma_x, sigma_y]
            ).permute(0, 2, 3, 1)

        color = color * opacity + background_color * (1 - opacity)

    return color, opacity

def save_model(radiance_field: nn.Module, path: Path):
    print(f"Saving model to {path.absolute()}")
    torch.save(radiance_field.state_dict(), path)

def main():
    args = get_args()

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # Setup neural network blocks
    from radiance_fields.ngp import NGPradianceField
    radiance_field = NGPradianceField(
        aabb=args.aabb,
        unbounded=args.unbounded,
    ).to(device)
    # Load pretrained weights
    if args.load_model_path is not None:
        radiance_field.load_state_dict(torch.load(args.load_model_path, map_location=device))

    # Optimizer only on 3D object
    optimizer = torch.optim.Adam(
        radiance_field.parameters(), lr=args.learning_rate
    )

    # Image and text scorer
    text_image_discriminator = TextImageDiscriminator().to(device)
    # Freeze all CLIP weights.
    text_image_discriminator.eval()
    for name, params in text_image_discriminator.named_parameters():
        params.requires_grad = False

    # Mechanism in order to have sparsity: Saves a bunch of compute
    occupancy_grid = OccupancyGrid(
        roi_aabb=args.aabb,
        resolution=args.grid_resolution,
        contraction_type=ContractionType.AABB,
    ).to(device)

    # Precompute all text embeddings
    prompter = ViewDependentPrompter(args.text)
    all_texts = prompter.get_all_text_prompts()
    with torch.no_grad():
        text_to_encodings = {
            text: text_image_discriminator.encode_texts([text])[0]
            for text in all_texts
        }

    # background probabilities
    training_background_probs = {
        Background.RANDOM_COLOR_UNIFORM_BACKGROUND: 1,
        Background.RANDOM_COLOR_BACKGROUND: 1,
        Background.CHECKERBOARD: 1,
        # Should already be handled by `Background.RANDOM_COLOR_UNIFORM_BACKGROUND`
        # Background.WHITE: 1,
        # Background.BLACK: 1
        # TODO @thomasw21: Implement
        # Background.RANDOM_TEXTURE: 1,
    }
    training_backgrounds = list(training_background_probs.keys())


    # training
    start_time = time.time()
    nb_iterations_for_time_estimation = 0
    for it in range(args.iterations):
        nb_iterations_for_time_estimation += 1

        # Set radiance field to trainable
        radiance_field.train()

        # generate a random camera view
        image_height, image_width = text_image_discriminator.image_height_width

        phis = generate_random_360_angles(num_angles=args.batch_size, device=device)

        # Generate a view dependent prompt
        encoded_texts = torch.stack([
            text_to_encodings[prompter.get_camera_view_prompt(phi)]
            for phi in phis]
        )

        # Update sparse occupancy matrix every n steps
        # Essentially there's a bunch of values that I don't care about since I can just set them to zero once and for all
        if it % args.update_occupancy_grid_interval == 0:
            # TODO @thomasw21: we're not using their official API, though I'm more than okay with this
            occupancy_grid._update(
                step=it,
                # TODO @thomasw21: figure out what the correct step_size we use.
                occ_eval_fn=lambda x: radiance_field.query_density(x) * 1e-3,
                occ_thre=0.01,
                # at which point we consider, it proposes a weird binary thing where you're higher than the mean clamped with this threshold
                ema_decay=0.95,  # exponential decay in order to create some sort of inertia
                warmup_steps=256,  # after which we sample randomly the grid for some values
            )

        # Render image
        sensors = generate_sensors(
            image_height=image_height,
            image_width=image_width,
            theta=args.training_theta,
            phis=phis
        )
        images, opacities = render_images(
            radiance_field,
            query_density=radiance_field.query_density,
            occupancy_grid=occupancy_grid,
            image_height=image_height,
            image_width=image_width,
            sensors=sensors,
        )

        # Augment images
        # TODO @thomasw21 change background for each image and not for each batch
        images, opacities = data_augment(images, opacities, background=choices(training_backgrounds, weights=[training_background_probs[bkd] for bkd in training_backgrounds])[0])

        # Discriminate images with text
        images = images.permute(0, 3, 1, 2) # [B, H, W, C] -> [B, C, H, W]
        encoded_images = text_image_discriminator.encode_images(images, encoded_texts=encoded_texts)
        scores = text_image_discriminator(encoded_images=encoded_images, encoded_texts=encoded_texts)
        mean_score = scores.mean()

        sublosses = [- mean_score]
        if args.lambda_transmittance_loss > 0:
            sublosses.append(
                - args.lambda_transmittance_loss * torch.clamp(1 - opacities.mean(), max=args.transmittance_loss_ceil)
            )

        # Compute entropy
        if args.lambda_transmittance_entropy > 0:
            clamped_opacities = torch.clamp(opacities, min=1e-5, max=1 - 1e-5)
            minus_clamped_opacities = 1 - clamped_opacities
            entropy = torch.mean( - clamped_opacities * torch.log(clamped_opacities) - minus_clamped_opacities * torch.log(minus_clamped_opacities))
            sublosses.append(
                args.lambda_transmittance_entropy * entropy
            )

        # Compute loss
        loss = sum(sublosses)

        # Optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Log loss
        if it % args.log_interval == 0:
            print(
                f"iteration: {it}/{args.iterations}| "
                f"time per iteration: {((time.time() - start_time) / nb_iterations_for_time_estimation):2f} sec / it | "
                f"loss: {loss.detach():6f} | "
                f"text/image score: {mean_score.detach():6f} | "
                f"opacity: {opacities.detach().mean():6f} | "
                f"{f'entropy: {entropy:6f} | ' if args.lambda_transmittance_entropy > 0 else ''}"
            )
            nb_iterations_for_time_estimation = 0
            start_time = time.time()

    # Save path
    save_model(radiance_field, args.save_model_path)

    # Generate some sample images
    radiance_field.eval()
    with torch.no_grad():
        phis = generate_random_360_angles(32, device=device, stochastic_angles=False)
        # TODO @thomasw21: define resolution
        R, C, K = generate_sensors(
            image_height=256,
            image_width=256,
            theta=args.validation_theta,
            phis=phis
        )
        images, opacities = render_images(
            radiance_field,
            query_density=radiance_field.query_density,
            # TODO @thomasw21: Check if I should actually feed the `occupancy_grid` or just rely on `query_density`
            occupancy_grid=occupancy_grid,
            image_height=256,
            image_width=256,
            sensors=(R, C, K),
        )
        print(f"Saving images to {args.save_images_path.absolute()}")
        save_image(
            tensor=images.permute(0, 3, 1, 2), #channel first
            fp=args.save_images_path,
        )
        images, opacities = data_augment(color=images, opacity=opacities, background=Background.WHITE)
        save_image(
            tensor=images.permute(0, 3, 1, 2),
            fp=args.save_images_path.parent / f"{args.save_images_path.stem}_aug{args.save_images_path.suffix}"
        )


if __name__ == "__main__":
    main()