import argparse
import time
from typing import Tuple, List, Callable

import numpy as np
import torch
from torch import nn
from tqdm import tqdm
from transformers import CLIPModel, CLIPTokenizer, CLIPProcessor, AutoConfig

import nerfacc
from nerfacc import OccupancyGrid, ContractionType
from radiance_fields.ngp import NGPradianceField


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", type=str, required=True, help="Text to fit")

    ###
    parser.add_argument(
        "--aabb",
        type=lambda s: [float(item) for item in s.split(",")],
        default="-0.5,-0.5,-0.5,0.5,0.5,0.5",
        help="delimited list input",
    )
    parser.add_argument("--unbounded", action="store_true", help="whether to use unbounded rendering")
    parser.add_argument("--update-occupancy-grid-interval", type=int, default=16, help="Update occupancy grid every n steps")

    ### Optimizer
    parser.add_argument("--learning-rate", type=int, default=1e-2, help="Learning rate")
    parser.add_argument("--iterations", type=int, default=2000, help="Number of optimizer steps")
    parser.add_argument("--batch-size", type=int, default=32, help="Number of camera views within a single batch")

    ### Logging
    parser.add_argument("--log-interval", type=int, default=10, help="Log every n steps")

    args = parser.parse_args()
    args.grid_resolution = 128
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

def generate_random_360_angles(num_angles: int, stochastic_angles: bool = True) -> torch.Tensor:
    if stochastic_angles:
        return torch.rand(num_angles) * 360.0
    else:
        return torch.arange(num_angles) * (360.0 / num_angles)

Sensors = Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
def generate_sensors(
    image_height: int,
    image_width: int,
    angles: torch.Tensor
) -> Sensors:
    N, = angles.shape

    # Angle from equator, 30 degrees from equator
    theta = (90 - 30) * np.pi / 180
    # Rotate according to y-axis
    rot_theta = torch.tensor(
        [
            [np.cos(theta), 0, np.sin(theta)],
            [0, 1, 0],
            [-np.sin(theta), 0, np.cos(theta)],
        ],
        device=angles.device,
        dtype=angles.dtype
    ) # [3, 3]

    sin_angles = torch.sin(angles) # [N,]
    cos_angles = torch.cos(angles) # [N,]

    # Rotate according to z-axis
    rot_phi = torch.zeros(N, 3, 3, device=angles.device, dtype=angles.dtype)
    rot_phi[:, -1, -1] = 1
    rot_phi[:, 0, 0] = cos_angles
    rot_phi[:, 1, 0] = -sin_angles
    rot_phi[:, 0, 1] = sin_angles
    rot_phi[:, 1, 1] = cos_angles

    rotations = rot_phi @ rot_theta

    # Origins
    radius = 2
    origins = \
        rotations @ torch.tensor([0, 0, radius], device=angles.device, dtype=angles.dtype) \
        + torch.tensor([0.5, 0.5, 0.5], device=angles.device, dtype=angles.dtype) # origin of the bounding box is 0

    # Camera specific values
    camera_angle_x = 45
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
        device=angles.device,
        dtype=angles.dtype
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
        # TODO @thomasw21: Sample a lot more position than this
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
        # TODO @thomasw21: Sample a lot more position than this
        positions = t_origins + t_dirs * (t_starts + t_ends) / 2.0
        rgbs, sigmas = radiance_field(positions, condition=t_dirs)
        return rgbs, sigmas  # (n_samples, 3), (n_samples, 1)
    return rgb_sigma_fn


def render_images(
    radiance_field: nn.Module,
    query_density: Callable[[torch.Tensor], torch.Tensor],
    occupancy_grid: OccupancyGrid,
    image_height: int,
    image_width: int,
    sensors: Sensors,
    # scene_aabb: torch.Tensor
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
        camera_intrinsics[0, 0][None, None, None]
    ], dim=-1) # [H, W, 3]
    # TODO @thomasw21: determine if I should run contiguous here
    camera_rotations_inv = camera_rotations.permute(0,2,1)
    directions = (camera_rotations_inv[:, None, :, :] @ pixel_position_in_camera.view(1, image_height * image_width, 1, 3)).unsqueeze(-2) # [N, H, W, 3]
    # pixel_position_in_world = directions + C[:, None, None, :]
    view_dirs = directions / torch.linalg.norm(directions, dim=-1)[..., None].view(-1, 3) # [N, H, W, 3]
    # TODO @thomasw21: Figure out a way without copying data
    origins = torch.repeat_interleave(camera_centers, image_height * image_height, dim=0)

    #### From nerfacc README:
    # Efficient Raymarching: Skip empty and occluded space, pack samples from all rays.
    # packed_info: (n_rays, 2). t_starts: (n_samples, 1). t_ends: (n_samples, 1).
    packed_info, t_starts, t_ends = nerfacc.ray_marching(
        origins,
        view_dirs,
        sigma_fn=get_sigma_fn(query_density, rays_o=origins, rays_d=view_dirs),
        # scene_aabb=args.scene_aabb # TODO @thomasw21: Need to pass it down otherwise this is going to be hell
        grid=occupancy_grid,
        near_plane=0.2,
        far_plane=1.0,
        early_stop_eps=1e-4,
        alpha_thre=1e-2,
    )

    # Differentiable Volumetric Rendering.
    # colors: (n_rays, 3). opacity: (n_rays, 1). depth: (n_rays, 1).
    color, opacity, depth = nerfacc.rendering(
        rgb_sigma_fn=get_rgb_sigma_fn(radiance_field, rays_o=origins, rays_d=view_dirs),
        packed_info=packed_info,
        t_starts=t_starts,
        t_ends=t_ends
    )

    return color.view(N, image_height, image_width, 3), opacity.view(N, image_height, image_width, 3)


def main():
    args = get_args()

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    # Setup neural network blocks
    radiance_field = NGPradianceField(
        aabb=args.aabb,
        unbounded=args.unbounded,
    ).to(device)

    optimizer = torch.optim.Adam(
        radiance_field.parameters(), lr=args.learning_rate
    )

    text_image_discriminator = TextImageDiscriminator()
    # Freeze all CLIP weights.
    text_image_discriminator.eval()
    for name, params in text_image_discriminator.named_parameters():
        params.requires_grad = False

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

    # training
    tic = time.time()
    for it in tqdm(range(args.iterations)):
        # Set radiance field to trainable
        radiance_field.train()

        # generate a random camera view
        image_height, image_width = text_image_discriminator.image_height_width

        angles = generate_random_360_angles(num_angles=args.batch_size)

        # Generate a view dependent prompt
        encoded_texts = torch.stack([
            text_to_encodings[prompter.get_camera_view_prompt(angle)]
            for angle in angles]
        )

        # Render image
        sensors = generate_sensors(image_height=image_height, image_width=image_width, angles=angles)
        images, opacities = render_images(
            radiance_field,
            query_density=radiance_field.query_density,
            occupancy_grid=occupancy_grid,
            image_height=image_height,
            image_width=image_width,
            sensors=sensors
        )

        # Discriminate images with text
        encoded_images = text_image_discriminator.encode_images(images, encoded_texts=encoded_texts)
        scores = text_image_discriminator(encoded_images=encoded_images, encoded_texts=encoded_texts)
        cosine_loss = - scores.mean()

        # Compute loss
        sublosses = [
            cosine_loss
        ]
        loss = sum(sublosses)

        # Optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update sparse occupancy matrix every n steps
        # Essentially there's a bunch of values that I don't care about since I can just set them to zero once and for all
        if it % args.update_occupancy_grid_interval == 0:
            # TODO @thomasw21: we're not using their official API, though I'm more than okay with this
            occupancy_grid._update(
                step=it,
                occ_eval_fn=lambda x: radiance_field.query_density(x) * args.grid_resolution,
                occ_thre=0.01, # at which point we consider, it proposes a weird binary thing where you're higher than the mean clamped with this threshold
                ema_decay=0.95, # exponential decay in order to create some sort of inertia
                warmup_steps=256, # after which we sample randomly the grid for some values
            )

        # Log loss
        if it % args.log_interval == 0:
            # TODO @thomasw21: get a logging mechanism going
            raise NotImplementedError

    pass

if __name__ == "__main__":
    main()