import argparse
import time
from enum import Enum
from pathlib import Path
from random import choices
from typing import Tuple, List, Callable, Optional, Set, Any, Dict

import numpy as np
import torch
import torchvision.transforms
from torchvision.transforms import functional as F
from torch import nn
from torchvision.utils import save_image

import nerfacc
from background import MLPBackground
from text_image_discriminator.clip import CLIPTextImageDiscriminator
from text_image_discriminator.sd import SDTextImageDiscriminator
from nerfacc import (
    OccupancyGrid,
    ContractionType,
    unpack_info,
    render_visibility,
)

"""
TODOs:
 - Create a video
 - Learn background color instead for seperation for sd, it seems otherwise the model tries to fill it itself
 - Make faster training by allowing distributed training
"""


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", type=str, required=True, help="Text to fit")
    parser.add_argument(
        "--save-model-path",
        type=Path,
        required=True,
        help="Where to save the model",
    )
    parser.add_argument(
        "--load-model-path", type=Path, help="Where to load the model from"
    )
    parser.add_argument(
        "--save-images-path",
        type=Path,
        required=True,
        help="Where to save the images that we generate at the end",
    )

    # Training nerf configs
    parser.add_argument(
        "--text-image-discriminator",
        type=str,
        choices=["clip", "sd"],
        help="Determine which text/image discriminator to use.",
    )
    parser.add_argument(
        "--aabb",
        type=lambda s: [float(item) for item in s.split(",")],
        default="-0.5,-0.5,-0.5,0.5,0.5,0.5",
        help="delimited list input",
    )
    parser.add_argument(
        "--background",
        type=str,
        choices=["random_backgrounds", "learned_background"],
        default="random_backgrounds",
    )
    parser.add_argument(
        "--unbounded",
        action="store_true",
        help="whether to use unbounded rendering",
    )
    parser.add_argument(
        "--update-occupancy-grid-interval",
        type=int,
        default=16,
        help="Update occupancy grid every n steps",
    )
    parser.add_argument("--ray-resample-in-training", action="store_true")
    parser.add_argument(
        "--stochastic-rays-through-pixels",
        action="store_true",
        help="Flag to allow model to sample any ray that goes through the pixel",
    )
    parser.add_argument(
        "--use-viewdirs",
        action="store_true",
        help="Whether the model use view dir in order to generate voxel color",
    )
    parser.add_argument(
        "--track-scene-origin-decay",
        type=float,
        default=0.999,
        help="Track scene origin with decay",
    )
    parser.add_argument("--use-occupancy-grid", action="store_true")
    parser.add_argument(
        "--training-thetas",
        type=lambda x: tuple(float(elt) for elt in x.split(",")),
        default=[60, 90],
        help="Elevation angle you're training at",
    )
    parser.add_argument(
        "--training-phis",
        type=lambda x: tuple(float(elt) for elt in x.split(",")),
        default=[0, 360],
        help="Around the lattitude you're training at",
    )
    parser.add_argument(
        "--training-resolution",
        type=lambda x: tuple(int(elt) for elt in x.split(",")),
        default=[224, 224],
        help="Image resolution to generate",
    )
    parser.add_argument(
        "--validation-thetas",
        type=lambda x: tuple(float(elt) for elt in x.split(",")),
        default=[45, 45],
        help="Elevation angle you're validatin at",
    )
    parser.add_argument(
        "--validation-phis",
        type=lambda x: tuple(float(elt) for elt in x.split(",")),
        default=[0, 360],
        help="Around the lattitude you're validating at",
    )
    parser.add_argument(
        "--validation-resolution",
        type=lambda x: tuple(int(elt) for elt in x.split(",")),
        default=[256, 256],
        help="Image resolution to generate",
    )

    # Optimizer
    # See DreamFields paper
    parser.add_argument("--lambda-transmittance-loss", type=float, default=0.5)
    parser.add_argument(
        "--transmittance-loss-ceil-range",
        type=lambda x: tuple(float(elt) for elt in x.split(",")),
        default=(0.5, 0.9),
    )
    parser.add_argument(
        "--transmittance-loss-ceil-exponential-annealing-step",
        type=int,
        default=500,
    )
    # DreamFusion paper loss
    parser.add_argument("--lambda-opacity", type=float, default=1e-3)
    # Dreamfusion Open source implementation
    parser.add_argument(
        "--lambda-transmittance-entropy", type=float, default=1e-4
    )
    # Center loss, for all the sigmas to be close to 0
    parser.add_argument("--lambda-center-loss", type=float, default=0.0)
    parser.add_argument(
        "--learning-rate", type=float, default=1e-2, help="Learning rate"
    )
    parser.add_argument(
        "--iterations", type=int, default=2000, help="Number of optimizer steps"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Number of camera views within a single batch",
    )

    # Logging
    parser.add_argument(
        "--log-interval", type=int, default=100, help="Log every n steps"
    )
    parser.add_argument(
        "--validation-interval",
        type=int,
        default=100,
        help="Log validation every n steps",
    )

    args = parser.parse_args()
    if args.use_occupancy_grid:
        assert (
            False
        ), "TODO @thomasw21: Figure out show to handle that grid resolution, with image resolution with rendering steps ..."
        args.grid_resolution = 128

    args.save_model_path.parent.mkdir(parents=True, exist_ok=True)
    args.save_images_path.parent.mkdir(parents=True, exist_ok=True)

    return args


class ViewDependentPrompter:
    """Update input text to condition on camera view"""

    overhead_suffix = "overhead view"
    bottom_suffix = "bottom view"
    phi_suffixes = ["front view", "side view", "back view", "side view"]

    def __init__(self, text: str):
        self.text = text

    def get_camera_view_prompt(self, theta: float, phi: float) -> str:
        """Given a homogenous matrix, we return the updated prompt"""
        # TODO @thomasw21: code a generic one using sensor instead of the angles

        if theta < 30:
            return self.get_prompt(self.overhead_suffix)

        if theta > 120:
            return self.get_prompt(self.bottom_suffix)

        quadrants = [60, 180, 240, 360]
        assert len(self.phi_suffixes) == len(quadrants)
        assert quadrants[-1] == 360
        for suffix, angle_limit in zip(self.phi_suffixes, quadrants):
            if phi <= angle_limit:
                return self.get_prompt(suffix)

    def get_prompt(self, suffix: str):
        return f"{self.text}, {suffix}"

    def get_all_text_prompts(self) -> Set[str]:
        return set(
            self.get_prompt(suffix)
            for suffix in set(
                (self.phi_suffixes + [self.overhead_suffix, self.bottom_suffix])
            )
        )


@torch.no_grad()
def generate_random_views(
        num_views: int,
        device: torch.device,
        theta_range: Tuple[float, float],
        phi_range: Tuple[float, float],
        stochastic_views: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # TODO @thomasw21: Interestingly this doesn't sample uniformily around the cirle (typically upsample regions around the poles)
    #  - To sample around the circle you need use normalized gaussian but otherwise I need to figure things out.
    min_theta, max_theta = theta_range
    min_phi, max_phi = phi_range
    if stochastic_views:
        # TODO @thomasw21: Activate uniform sampling over sphere surface
        # if np.random.rand() > 0.5:
        if False:
            # biased sampling. More sampling towards the top view
            thetas = (
                    torch.rand(num_views, device=device) * (max_theta - min_theta)
                    + min_theta
            )
            phis = (
                    torch.rand(num_views, device=device) * (max_phi - min_phi)
                    + min_phi
            )
        else:
            # http://corysimon.github.io/articles/uniformdistn-on-sphere/
            cos_min_theta = np.cos(min_theta * torch.pi / 180)
            cos_max_theta = np.cos(max_theta * torch.pi / 180)
            thetas = (
                    torch.acos(
                        torch.rand(num_views, device=device)
                        * (cos_max_theta - cos_min_theta)
                        + cos_min_theta
                    )
                    * 180
                    / torch.pi
            )
            phis = (
                    torch.rand(num_views, device=device) * (max_phi - min_phi)
                    + min_phi
            )
    else:
        assert num_views % 2 == 0
        half_num_views = num_views // 2
        if max_theta == min_theta:
            thetas = torch.full((num_views,), min_theta, device=device)
        else:
            # We're going to make two full circles one descending on going up
            thetas = torch.cat(
                [
                    torch.arange(
                        min_theta,
                        max_theta,
                        step=(max_theta - min_theta) / half_num_views,
                        device=device,
                    ),
                    torch.arange(
                        max_theta,
                        min_theta,
                        step=(min_theta - max_theta) / half_num_views,
                        device=device,
                    ),
                ]
            )

        if min_phi == max_phi:
            phis = torch.full((num_views,), min_phi, device=device)
        else:
            # We're going to make two full circles one descending on going up
            phis = torch.cat(
                [
                    torch.arange(
                        min_phi,
                        max_phi,
                        step=(max_phi - min_phi) / half_num_views,
                        device=device,
                    ),
                    torch.arange(
                        min_phi,
                        max_phi,
                        step=(max_phi - min_phi) / half_num_views,
                        device=device,
                    ),
                ]
            )

    # Radius
    if stochastic_views:
        ratio_with_dream_fusion = np.sqrt(3 * 0.5 ** 2) / 1.4  # DreamFusion
        radius = (
                         torch.rand(num_views, device=device) * 0.5 + 1
                 ) * ratio_with_dream_fusion
    else:
        radius = torch.tensor(1, device=device, dtype=torch.float)[None].expand(
            num_views
        )
    return thetas, phis, radius


Sensors = Tuple[torch.Tensor, torch.Tensor, torch.Tensor]


@torch.no_grad()
def generate_sensors(
    image_height: int,
    image_width: int,
    thetas: torch.Tensor,
    phis: torch.Tensor,
    radius: torch.Tensor,
    scene_origin: Optional[torch.Tensor] = None,
) -> Sensors:
    (N,) = phis.shape
    device = phis.device
    dtype = phis.dtype

    # Angle from equator, 30 degrees from equator
    thetas = thetas * np.pi / 180
    sin_thetas = torch.sin(thetas)
    cos_thetas = torch.cos(thetas)
    rot_theta = torch.zeros(N, 3, 3, device=device, dtype=dtype)
    rot_theta[:, 1, 1] = 1
    rot_theta[:, 0, 0] = cos_thetas
    rot_theta[:, 2, 0] = -sin_thetas
    rot_theta[:, 0, 2] = sin_thetas
    rot_theta[:, 2, 2] = cos_thetas

    phis = phis * np.pi / 180
    sin_phis = torch.sin(phis)  # [N,]
    cos_phis = torch.cos(phis)  # [N,]

    # Rotate according to z-axis
    rot_phi = torch.zeros(N, 3, 3, device=device, dtype=dtype)
    rot_phi[:, -1, -1] = 1
    rot_phi[:, 0, 0] = cos_phis
    rot_phi[:, 1, 0] = sin_phis
    rot_phi[:, 0, 1] = -sin_phis
    rot_phi[:, 1, 1] = cos_phis

    rotations = rot_phi @ rot_theta  # [N, 3, 3]

    # Origins
    z = torch.stack(
        [
            torch.tensor(0)[None].expand(N),
            torch.tensor(0)[None].expand(N),
            radius,
        ],
        dim=-1,
    ).to(device=device, dtype=dtype)
    origins = torch.bmm(rotations, z[:, :, None]).squeeze(-1)  # [N, 3]
    # + torch.tensor([0.5, 0.5, 0.5], device=angles.device, dtype=angles.dtype) # origin of the bounding box is 0

    if scene_origin is not None:
        # origins was estimated to be in the center, we now shift it to be pointing towards the center of the scene
        origins = origins + scene_origin[None, :]

    # Camera specific values
    # TODO @thomasw21: define angle
    camera_angle_x = 45 * np.pi / 180
    ratio = (
            image_width / image_height
    )  # make sure that the pixels are actually square
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
        dtype=dtype,
    )

    return rotations, origins, K


# Rendering helpers
def get_sigma_fn(
    query_density: Callable[[torch.Tensor], torch.Tensor],
    rays_o: torch.Tensor,
    rays_d: torch.Tensor,
    perturb: bool,
) -> Callable[
    [torch.Tensor, torch.Tensor, torch.Tensor],
    Tuple[torch.Tensor, torch.Tensor],
]:
    def sigma_fn(
            t_starts: torch.Tensor, t_ends: torch.Tensor, ray_indices: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Query density values from a user-defined radiance field.
        :params t_starts: Start of the sample interval along the ray. (n_samples, 1).
        :params t_ends: End of the sample interval along the ray. (n_samples, 1).
        :params ray_indices: Ray indices that each sample belongs to. (n_samples,).
        :returns The post-activation density values. (n_samples, 1).
        """

        t_origins = rays_o[ray_indices]  # (n_samples, 3)
        t_dirs = rays_d[ray_indices]  # (n_samples, 3)
        # This essentially compute the mean position, we might instead jitter and randomly sample for robustness
        if perturb:
            random_t = (
                    torch.rand_like(t_starts) * (t_ends - t_starts) + t_starts
            )
            positions = t_origins + t_dirs * random_t
        else:
            positions = t_origins + t_dirs * (t_starts + t_ends) / 2.0
        sigmas = query_density(positions)
        return sigmas, positions  # (n_samples, 1)

    return sigma_fn


def get_rgb_sigma_fn(
    radiance_field: nn.Module,
    rays_o: torch.Tensor,
    rays_d: torch.Tensor,
    perturb: bool,
) -> Callable[
    [torch.Tensor, torch.Tensor, torch.Tensor],
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
]:
    def rgb_sigma_fn(
            t_starts: torch.Tensor, t_ends: torch.Tensor, ray_indices: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Query rgb and density values from a user-defined radiance field.
        :params t_starts: Start of the sample interval along the ray. (n_samples, 1).
        :params t_ends: End of the sample interval along the ray. (n_samples, 1).
        :params ray_indices: Ray indices that each sample belongs to. (n_samples,).
        :returns The post-activation rgb and density values.
            (n_samples, 3), (n_samples, 1).
        """
        t_origins = rays_o[ray_indices]  # (n_samples, 3)
        t_dirs = rays_d[ray_indices]  # (n_samples, 3)
        # This essentially compute the mean position, we might instead jitter and randomly sample for robustness
        if perturb:
            random_t = (
                    torch.rand_like(t_starts) * (t_ends - t_starts) + t_starts
            )
            positions = t_origins + t_dirs * random_t
        else:
            positions = t_origins + t_dirs * (t_starts + t_ends) / 2.0
        rgbs, sigmas = radiance_field(positions, t_dirs)
        return rgbs, sigmas, positions  # (n_samples, 3), (n_samples, 1)

    return rgb_sigma_fn

Rays = Tuple[torch.Tensor, torch.Tensor]
def render_images(
    radiance_field: nn.Module,
    query_density: Callable[[torch.Tensor], torch.Tensor],
    occupancy_grid: Optional[OccupancyGrid],
    aabb: Optional[torch.Tensor],
    image_height: int,
    image_width: int,
    sensors: Sensors,
    ray_resample: bool = False,
    stochastic_rays_through_pixels: bool = False,
    stratified: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Rays]:
    device = sensors[0].device
    camera_rotations, camera_centers, camera_intrinsics = sensors
    N = camera_rotations.shape[0]
    if occupancy_grid is not None:
        assert aabb is None
        aabb = occupancy_grid.roi_aabb
    else:
        assert aabb is not None

    # Compute which rays we should run
    # # We for now compute the center of each pixel, thus the 0.5
    # # Get help: https://www.cs.cmu.edu/~16385/s17/Slides/11.1_Camera_matrix.pdf
    assert (
            camera_intrinsics[0, 0] == camera_intrinsics[1, 1]
    ), "focal needs to be the same in x and y"
    x, y = torch.meshgrid(
        torch.arange(image_height, device=device)
        - camera_intrinsics[0, 2]
        + 0.5,
        torch.arange(image_width, device=device)
        - camera_intrinsics[1, 2]
        + 0.5,
        indexing="ij",
    )  # [H, W]
    # TODO @thomasw21: sample from gaussian instead though it probably needs to be bounded.
    if stochastic_rays_through_pixels:
        x = x + torch.rand_like(x) - 0.5
        y = y + torch.rand_like(y) - 0.5
    pixel_position_in_camera = (
            torch.stack(
                [
                    x,
                    y,
                    -camera_intrinsics[0, 0][None, None].expand(
                        image_height, image_height
                    ),
                ],
                dim=-1,
            )
            / camera_intrinsics[0, 0]
    )  # [H, W, 3]
    # Assume that the cameras are looking at the origin 0.
    directions = (
            camera_rotations[:, None, :, :]
            @ pixel_position_in_camera.view(1, image_height * image_width, 3, 1)
    ).squeeze(
        -1
    )  # [N, H, W, 3]
    # pixel_position_in_world = directions + C[:, None, None, :]
    view_dirs = (
            directions / torch.linalg.norm(directions, dim=-1)[..., None]
    ).view(
        -1, 3
    )  # [N * image_height * image_width, 3]
    # TODO @thomasw21: Figure out a way without copying data
    origins = torch.repeat_interleave(
        camera_centers, image_height * image_height, dim=0
    )

    #### From nerfacc README:
    # Efficient Raymarching: Skip empty and occluded space, pack samples from all rays.
    # packed_info: (n_rays, 2). t_starts: (n_samples, 1). t_ends: (n_samples, 1).
    # TODO @thomasw21: determine the chunk size
    chunk = 2 ** 15
    results = []
    sum_sigmas = 0
    numerator = torch.tensor([0.0, 0.0, 0.0], device=device, dtype=torch.float)
    for i in range(0, N * image_height * image_width, chunk):
        origins_shard = origins[i: i + chunk]
        view_dirs_shard = view_dirs[i: i + chunk]
        with torch.no_grad():
            packed_info, t_starts, t_ends = nerfacc.ray_marching(
                rays_o=origins_shard,
                rays_d=view_dirs_shard,
                sigma_fn=None,
                scene_aabb=aabb,  # TODO @thomasw21: Need to pass it down otherwise this is going to be hell
                grid=occupancy_grid,  # This is fucked
                # near_plane=0.2,
                # far_plane=1.0,
                early_stop_eps=0.0,
                # alpha_thre=1e-4, # nerfstudio uses 1e-4, default is 0.0
                stratified=False,
            )

            sigma_fn = get_sigma_fn(
                query_density,
                rays_o=origins_shard,
                rays_d=view_dirs_shard,
                perturb=stratified,
            )
            if ray_resample:
                # Select only visible segments
                # Query sigma without gradients
                ray_indices = unpack_info(packed_info)
                sigmas, positions = sigma_fn(
                    t_starts, t_ends, ray_indices.long()
                )
                weights = nerfacc.render_weight_from_density(
                    packed_info=packed_info,
                    t_starts=t_starts,
                    t_ends=t_ends,
                    sigmas=sigmas,
                )
                packed_info, t_starts, t_ends = nerfacc.ray_resampling(
                    packed_info=packed_info,
                    t_starts=t_starts,
                    t_ends=t_ends,
                    weights=weights,
                    n_samples=196,
                )
            else:

                # Select only visible segments
                # Query sigma without gradients
                ray_indices = unpack_info(packed_info)
                sigmas, positions = sigma_fn(
                    t_starts, t_ends, ray_indices.long()
                )
                assert (
                        sigmas.shape == t_starts.shape
                ), "sigmas must have shape of (N, 1)! Got {}".format(
                    sigmas.shape
                )
                alphas = 1.0 - torch.exp(-sigmas * (t_ends - t_starts))

                # Compute visibility of the samples, and filter out invisible samples
                visibility, packed_info_visible = render_visibility(
                    packed_info,
                    alphas=alphas,
                    early_stop_eps=0.0,
                    alpha_thre=1e-4,
                )
                t_starts, t_ends = t_starts[visibility], t_ends[visibility]
                packed_info = packed_info_visible

            # Estimated mean position
            # TODO @thomasw21: I might need to device by len(sigmas)
            # TODO @thomasw21: this estimation might compute mutliple times the values we deem visible, which might force the estimation to oversample
            sum_sigmas += torch.sum(sigmas)
            numerator += torch.sum(positions * sigmas, dim=0)

        # Differentiable Volumetric Rendering.
        # colors: (n_rays, 3). opacity: (n_rays, 1). depth: (n_rays, 1).
        rgb_sigma_fn = get_rgb_sigma_fn(
            radiance_field,
            rays_o=origins_shard,
            rays_d=view_dirs_shard,
            perturb=stratified,
        )
        color, opacity, depth = nerfacc.rendering(
            rgb_sigma_fn=lambda *args: rgb_sigma_fn(*args)[:2],
            packed_info=packed_info,
            t_starts=t_starts,
            t_ends=t_ends,
        )
        results.append((color, opacity, depth))

        # Compute sigma loss to force the model to be at the center

    color = torch.cat([elt[0] for elt in results], dim=0).view(
        N, image_height, image_width, 3
    )
    opacity = torch.cat([elt[1] for elt in results], dim=0).view(
        N, image_height, image_width, 1
    )

    density_origin = numerator / sum_sigmas
    return color, opacity, density_origin, (origins, view_dirs)


class Background(Enum):
    RANDOM_COLOR_UNIFORM_BACKGROUND = 1
    RANDOM_COLOR_BACKGROUND = 2
    RANDOM_TEXTURE = 3
    CHECKERBOARD = 4
    WHITE = 5
    BLACK = 6
    LEARNED = 7


def add_background(
    color: torch.Tensor,
    opacity: torch.Tensor,
    rays: Rays,
    backgrounds: List[Optional[Tuple[Background, Any]]],
    blur_background: bool = True,
):
    N, H, W, _ = color.shape

    # Background
    background_colors = torch.empty_like(color)
    for i, background in enumerate(backgrounds):
        enum, value = background
        # Single colored background
        if background is None:
            # Hack as it shouldn't change anything
            background_color = color[i]
        elif enum is Background.RANDOM_COLOR_UNIFORM_BACKGROUND:
            background_color = torch.rand(1, 1, 3, device=color.device)
        elif enum is Background.RANDOM_COLOR_BACKGROUND:
            background_color = torch.rand(H, W, 3, device=color.device)
        elif enum is Background.RANDOM_TEXTURE:
            raise NotImplementedError
        elif enum is Background.CHECKERBOARD:
            # https://github.com/google-research/google-research/blob/4f54cade26f40728be7fda05c89011f89b7b7b7f/dreamfields/experiments/diffusion_3d/augment.py#L40
            nsq_x = 8
            nsq_y = 8
            assert H % nsq_x == 0
            assert W % nsq_y == 0
            sq_x = H // nsq_x
            sq_y = W // nsq_y
            color1, color2 = torch.rand(2, 3, device=color.device)
            background_color = (
                color1[None, None, :]
                .repeat(H, W, 1)
                .view(nsq_x, sq_x, nsq_y, sq_y, 3)
            )
            background_color[::2, :, 1::2, :, :] = color2[
                                                   None, None, None, None, :
                                                   ]
            background_color[1::2, :, ::2, :, :] = color2[
                                                   None, None, None, None, :
                                                   ]
            background_color = background_color.view(H, W, 3)
        elif enum is Background.WHITE:
            background_color = torch.ones(1, 1, 1, device=color.device)
        elif enum is Background.BLACK:
            background_color = torch.zeros(1, 1, 1, device=color.device)
        elif enum is Background.LEARNED:
            _, view_dirs = rays
            # value is assumed to be a MLPBackground
            assert isinstance(value, MLPBackground)
            background_color = value(view_dirs)
            assert background_color.shape[-1] == 3
            background_color = background_color.view(H, W, 3)
        else:
            raise ValueError

        if blur_background:
            min_blur, max_blur = (0.0, 10.0)
            sigma_x, sigma_y = (
                    np.random.rand(2) * (max_blur - min_blur) + min_blur
            )
            background_color = F.gaussian_blur(
                torch.broadcast_to(background_color, (H, W, 3)).permute(
                    2, 0, 1
                ),
                kernel_size=[15, 15],
                # Weird, but it's in dreamfields https://github.com/google-research/google-research/blob/00392d6e3bd30bfe706859287035fcd8d53a010b/dreamfields/dreamfields/config/config_base.py#L130
                sigma=[sigma_x, sigma_y],
            ).permute(1, 2, 0)

        background_colors[i] = background_color

    return color * opacity + background_colors * (1 - opacity)


def data_augment(
    color: torch.Tensor,
    opacity: torch.Tensor,
    rays: Rays,
    resize_shape: Tuple[int, int],
    random_resize_crop: bool,
    # TODO @thomasw21: Make random background color accessible through CLI
    backgrounds: Optional[List[Optional[Tuple[Background, Any]]]],
    blur_background: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    N, H, W, _ = color.shape
    # Do random crop
    img = torch.cat([color.permute(3, 0, 1, 2), opacity.permute(3, 0, 1, 2)])
    if random_resize_crop:
        transforms = torchvision.transforms.Compose(
            [
                # DreamFields
                # torchvision.transforms.RandomResizedCrop(size=resize_shape, scale=(0.80, 1.0))
                torchvision.transforms.Resize(size=resize_shape)
            ]
        )
    else:
        transforms = torchvision.transforms.Compose(
            [
                # DreamFields
                torchvision.transforms.Resize(size=resize_shape)
            ]
        )
    img = transforms(img)
    assert color.shape[-1] == 3
    assert opacity.shape[-1] == 1
    color = img[:3].permute(1, 2, 3, 0)
    opacity = img[-1:].permute(1, 2, 3, 0)

    if backgrounds is not None:
        color = add_background(
            color=color,
            opacity=opacity,
            rays=rays,
            backgrounds=backgrounds,
            blur_background=blur_background,
        )

    return color, opacity


def save_model(radiance_field: nn.Module, path: Path):
    print(f"Saving model to {path.absolute()}")
    torch.save(radiance_field.state_dict(), path)


def save_images(path: Path, images: torch.Tensor, opacities: torch.Tensor, rays: Rays):
    """For each image save one with black background and the other one with white background"""
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Saving images to {path.absolute()}")
    save_image(
        tensor=images.permute(0, 3, 1, 2),  # channel first
        fp=path,
    )
    images = add_background(
        color=images,
        opacity=opacities,
        rays=rays,
        backgrounds=[Background.WHITE for _ in range(len(images))],
        blur_background=False,
    )
    save_image(
        tensor=images.permute(0, 3, 1, 2),
        fp=path.parent / f"{path.stem}_white{path.suffix}",
    )


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
        use_viewdirs=args.use_viewdirs,
        # original_sigma_offset=1, # This is in order to force the model to be opaque at init, if I could I would update the weight directl
        original_sigma_offset=-1,
        spatial_density_bias=True,
    ).to(device)
    # Load pretrained weights
    if args.load_model_path is not None:
        radiance_field.load_state_dict(
            torch.load(args.load_model_path, map_location=device)
        )

    # Optimizer only on 3D object
    # TODO @thomasw21: Determine if we run tcnn.optimizers
    grad_scaler = torch.cuda.amp.GradScaler()

    optimizer = torch.optim.AdamW(radiance_field.get_params(args.learning_rate))
    # optimizer = torch.optim.Adam(radiance_field.get_params(args.learning_rate))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda iter: 0.1 ** min(iter / args.iterations, 1)
    )

    # Image and text scorer
    if args.text_image_discriminator == "clip":
        text_image_discriminator = CLIPTextImageDiscriminator().to(device)
    elif args.text_image_discriminator == "sd":
        text_image_discriminator = SDTextImageDiscriminator().to(device)
    else:
        raise ValueError(
            f"Invalid choice of text image disciminator. Got: {args.text_image_discriminator}"
        )
    # Freeze all CLIP weights.
    text_image_discriminator.eval()
    for name, params in text_image_discriminator.named_parameters():
        params.requires_grad = False

    # Mechanism in order to have sparsity: Saves a bunch of compute
    if args.use_occupancy_grid:
        occupancy_grid = OccupancyGrid(
            roi_aabb=args.aabb,
            resolution=args.grid_resolution,
            contraction_type=ContractionType.AABB,
        ).to(device)
        aabb = None  # It's already stored in `occupancy_grid`
    else:
        occupancy_grid = None
        aabb = torch.tensor(args.aabb, device=device)
    scene_origin = torch.tensor([0.0, 0.0, 0.0], device="cpu")

    # Precompute all text embeddings
    prompter = ViewDependentPrompter(args.text)
    all_texts = prompter.get_all_text_prompts()
    with torch.no_grad():
        text_to_encodings = {
            text: text_image_discriminator.encode_texts([text])
            for text in all_texts
        }

    # background
    # Dictionary, keys: background name, values: (probability, additional values)
    training_backgrounds: Dict[Background, Tuple[int, Any]]
    if args.background == "random_backgrounds":
        training_backgrounds = {
            Background.RANDOM_COLOR_UNIFORM_BACKGROUND: (1, None),
            Background.RANDOM_COLOR_BACKGROUND: (1, None),
            Background.CHECKERBOARD: (1, None),
            # Should already be handled by `Background.RANDOM_COLOR_UNIFORM_BACKGROUND`
            # Background.WHITE: (1,None),
            # Background.BLACK: (1,None)
            # TODO @thomasw21: Implement
            # Background.RANDOM_TEXTURE: (1,None),
        }
    elif args.background == "learned_background":
        training_backgrounds = {Background.LEARNED: (1, MLPBackground().to(device))}
    else:
        raise ValueError(f"Invalid choice of background. Got {args.background}")

    # training
    start_time = time.time()
    nb_iterations_for_time_estimation = 0
    for it in range(args.iterations):
        nb_iterations_for_time_estimation += 1

        # Set radiance field to trainable
        radiance_field.train()

        # generate a random camera view
        # training_image_height, training_image_width = text_image_discriminator.image_height_width
        training_image_height, training_image_width = args.training_resolution

        with torch.no_grad():
            thetas, phis, radius = generate_random_views(
                num_views=args.batch_size,
                device=torch.device("cpu"),  # no need to to everything in GPU
                theta_range=args.training_thetas,
                phi_range=args.training_phis,
                stochastic_views=True,
            )

            # Generate a view dependent prompt
            encoded_texts = torch.cat(
                [
                    text_to_encodings[
                        prompter.get_camera_view_prompt(theta=theta, phi=phi)
                    ]
                    for theta, phi in zip(thetas, phis)
                ]
            )

            sensors = generate_sensors(
                image_height=training_image_height,
                image_width=training_image_width,
                thetas=thetas,
                phis=phis,
                radius=radius,
                scene_origin=scene_origin,
            )
            sensors = tuple(elt.to(device) for elt in sensors)

        # Update sparse occupancy matrix every n steps
        # Essentially there's a bunch of values that I don't care about since I can just set them to zero once and for all
        # We update the scene origin as well
        if (
            args.use_occupancy_grid
            and args.update_occupancy_grid_interval != 0
            and it % args.update_occupancy_grid_interval == 0
        ):
            # TODO @thomasw21: we're not using their official API, though I'm more than okay with this
            occupancy_grid._update(
                step=it,
                # TODO @thomasw21: figure out what the correct step_size we use. Has to be something more or less proportional to the grid resolution
                occ_eval_fn=lambda x: radiance_field.query_density(x) * 1e-3,
                occ_thre=0.01,
                # at which point we consider, it proposes a weird binary thing where you're higher than the mean clamped with this threshold
                ema_decay=0.95,  # exponential decay in order to create some sort of inertia
                warmup_steps=256,  # after which we sample randomly the grid for some values
            )

        # Render image
        images, opacities, density_origin, rays = render_images(
            radiance_field,
            query_density=radiance_field.query_density,
            occupancy_grid=occupancy_grid,
            aabb=aabb,
            image_height=training_image_height,
            image_width=training_image_width,
            sensors=sensors,
            ray_resample=args.ray_resample_in_training,
            stochastic_rays_through_pixels=args.stochastic_rays_through_pixels,
            stratified=True,
        )

        # Augment images: we duplicate the rendered images
        images, opacities = data_augment(
            images,
            opacities,
            rays=rays,
            random_resize_crop=True,
            resize_shape=text_image_discriminator.image_height_width,
            backgrounds=choices(
                list((k, v[1]) for k, v in training_backgrounds.items()),
                weights=[
                    training_backgrounds[bkd][0] for bkd in training_backgrounds
                ],
                k=args.batch_size,
            ),
            blur_background=True,
        )

        # Discriminate images with text
        images = images.permute(0, 3, 1, 2)  # [B, H, W, C] -> [B, C, H, W]
        encoded_images = text_image_discriminator.encode_images(
            images, encoded_texts=encoded_texts
        )
        scores = text_image_discriminator(
            encoded_images=encoded_images, encoded_texts=encoded_texts
        )
        mean_score = scores.mean()

        ### Compute loss
        sublosses = [mean_score]
        if args.lambda_transmittance_loss > 0:
            t = min(
                it / args.transmittance_loss_ceil_exponential_annealing_step, 1
            )
            min_, max_ = args.transmittance_loss_ceil_range
            transmittance_loss_ceil = np.exp(
                np.log(min_) * (1 - t) + np.log(max_) * t
            )
            sublosses.append(
                -args.lambda_transmittance_loss
                * torch.mean(
                    torch.clamp(
                        1 - torch.mean(opacities, (1, 2)),
                        max=transmittance_loss_ceil,
                    )
                )
            )

        # Compute entropy
        # Close to https://cs.github.com/google-research/google-research/blob/219754bb2a058329174353efc7141434d1fd500e/dreamfields/dreamfields/lib.py#L173
        if args.lambda_transmittance_entropy > 0:
            clamped_opacities = torch.clamp(opacities, min=1e-6, max=1 - 1e-6)
            minus_clamped_opacities = 1 - clamped_opacities
            entropy = torch.mean(
                -clamped_opacities * torch.log(clamped_opacities)
                - minus_clamped_opacities * torch.log(minus_clamped_opacities)
            )
            sublosses.append(args.lambda_transmittance_entropy * entropy)

        if args.lambda_opacity > 0:
            sublosses.append(
                # args.lambda_opacity * torch.mean(torch.sqrt(opacities ** 2 + 0.01))
                args.lambda_opacity
                * torch.mean(opacities)
            )

        # Compute loss
        loss = sum(sublosses)

        # Optimizer step
        optimizer.zero_grad()
        grad_scaler.scale(loss).backward()
        grad_scaler.step(optimizer)
        scheduler.step()
        grad_scaler.update()

        # Update scene origin
        if args.track_scene_origin_decay < 1:
            scene_origin = (
                    args.track_scene_origin_decay * scene_origin.to("cpu")
                    + (1 - args.track_scene_origin_decay) * density_origin
            )

        # Log loss
        if it % args.log_interval == 0:
            print(
                f"iteration: {it}/{args.iterations}| "
                f"time per iteration: {((time.time() - start_time) / nb_iterations_for_time_estimation):2f} sec / it | "
                f"loss: {loss.detach():6f} | "
                f"text/image score: {mean_score.detach():6f} | "
                f"opacity: {opacities.detach().mean():6f} | "
                f"{f'entropy: {entropy:6f} | ' if args.lambda_transmittance_entropy > 0 else ''}"
                f"scene origin: {scene_origin} | "
            )
            nb_iterations_for_time_estimation = 0
            start_time = time.time()

        # Validation loss
        if it != 0 and it % args.validation_interval == 0:
            # TODO @thomasw21: FACTORISE THIS!!!
            radiance_field.eval()
            with torch.no_grad():
                # image_height, image_width = text_image_discriminator.image_height_width
                (
                    validation_image_height,
                    validation_image_width,
                ) = args.validation_resolution

                thetas, phis, radius = generate_random_views(
                    8,
                    device=torch.device("cpu"),
                    stochastic_views=False,
                    theta_range=args.validation_thetas,
                    phi_range=args.validation_phis,
                )

                sensors = generate_sensors(
                    image_height=validation_image_height,
                    image_width=validation_image_width,
                    thetas=thetas,
                    phis=phis,
                    radius=radius,
                    scene_origin=scene_origin,
                )
                sensors = tuple(elt.to(device) for elt in sensors)
                images, opacities, _, rays = render_images(
                    radiance_field,
                    query_density=radiance_field.query_density,
                    # TODO @thomasw21: Check if I should actually feed the `occupancy_grid` or just rely on `query_density`
                    occupancy_grid=occupancy_grid,
                    aabb=aabb,
                    image_height=validation_image_height,
                    image_width=validation_image_width,
                    sensors=sensors,
                    stochastic_rays_through_pixels=False,
                    ray_resample=False,
                    stratified=False,
                )

                # Resize image to correct shape
                images, opacities = data_augment(
                    images,
                    opacities,
                    rays=rays,
                    random_resize_crop=False,
                    resize_shape=text_image_discriminator.image_height_width,
                    backgrounds=None,
                    blur_background=False,
                )

                ### Compute loss
                channel_first_images = images.permute(0, 3, 1, 2)
                encoded_texts = torch.cat(
                    [
                        text_to_encodings[
                            prompter.get_camera_view_prompt(
                                theta=theta, phi=phi
                            )
                        ]
                        for theta, phi in zip(thetas, phis)
                    ]
                )
                encoded_images = text_image_discriminator.encode_images(
                    channel_first_images, encoded_texts=encoded_texts
                )

                chunk_size = 8
                scores = []
                text_image_ratio = len(encoded_texts) // len(encoded_images)
                for start in range(0, len(encoded_images), chunk_size):
                    end = start + chunk_size
                    encoded_images_chunk = encoded_images[start:end]
                    encoded_texts_chunk = encoded_texts[
                                          text_image_ratio * start: text_image_ratio * end
                                          ]
                    scores.append(
                        text_image_discriminator(
                            encoded_images=encoded_images_chunk,
                            encoded_texts=encoded_texts_chunk,
                        )
                    )
                mean_score = torch.cat(scores).mean()

                save_images(
                    images=images,
                    opacities=opacities,
                    rays=rays,
                    path=args.save_images_path
                         / "validation"
                         / f"{it}-of-{args.iterations}"
                         / "result.jpg",
                )

            print("##### validation")
            print(
                f"iteration: {it}/{args.iterations}| "
                f"text/image score: {mean_score.detach():6f} | "
                f"opacity: {opacities.detach().mean():6f} | "
            )
            print("#####")

    # Save path
    save_model(radiance_field, args.save_model_path)

    # Generate some sample images
    radiance_field.eval()
    with torch.no_grad():
        thetas, phis, radius = generate_random_views(
            32,
            device=torch.device("cpu"),
            stochastic_views=False,
            theta_range=args.validation_thetas,
            phi_range=args.validation_phis,
        )
        sensors = generate_sensors(
            image_height=args.validation_resolution[0],
            image_width=args.validation_resolution[1],
            thetas=thetas,
            phis=phis,
            radius=radius,
            scene_origin=scene_origin,
        )
        sensors = tuple(elt.to(device) for elt in sensors)
        images, opacities, _, rays= render_images(
            radiance_field,
            query_density=radiance_field.query_density,
            # TODO @thomasw21: Check if I should actually feed the `occupancy_grid` or just rely on `query_density`
            occupancy_grid=occupancy_grid,
            aabb=aabb,
            image_height=args.validation_resolution[0],
            image_width=args.validation_resolution[1],
            sensors=sensors,
            stochastic_rays_through_pixels=False,
            ray_resample=False,
            stratified=False,
        )

        ### Compute loss
        channel_first_images = images.permute(0, 3, 1, 2)
        resized_channel_first_images = F.resize(
            channel_first_images, text_image_discriminator.image_height_width
        )
        encoded_texts = torch.cat(
            [
                text_to_encodings[
                    prompter.get_camera_view_prompt(theta=theta, phi=phi)
                ]
                for theta, phi in zip(thetas, phis)
            ]
        )
        encoded_images = text_image_discriminator.encode_images(
            resized_channel_first_images, encoded_texts=encoded_texts
        )

        chunk_size = 8
        scores = []
        text_image_ratio = len(encoded_texts) // len(encoded_images)
        for start in range(0, len(encoded_images), chunk_size):
            end = start + chunk_size
            encoded_images_chunk = encoded_images[start:end]
            encoded_texts_chunk = encoded_texts[
                text_image_ratio * start: text_image_ratio * end
            ]
            scores.append(
                text_image_discriminator(
                    encoded_images=encoded_images_chunk,
                    encoded_texts=encoded_texts_chunk,
                )
            )
        mean_score = torch.cat(scores).mean()

        ### Logs
        print("###### Evaluation ##########")
        print(
            f"text/image score: {mean_score.detach():6f} | "
            f"opacity: {opacities.detach().mean():6f} | "
            f"scene origin: {scene_origin} | "
        )

        ## Saving
        save_images(
            images=images,
            opacities=opacities,
            rays=rays,
            path=args.save_images_path / "final.jpg",
        )


if __name__ == "__main__":
    main()
