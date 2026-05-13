from __future__ import annotations

import pathlib, json, math, math, time, random, os, copy


import numpy as np
import torch
import plyfile
import einops
import tqdm
import open3d as o3d

import torch.nn.functional as F

from libraries.configs import TracerConfig
from libraries.utilities import (
    ExLog,
    ExTimer,
    UTILITIES_IO,
    UTILITIES_IMAGE,
    UTILITY_FLIP,
    UTILITIES_COLOUR,
)


import eag_pt_tracer_optix

# [START: add backward]


def Differentiable_EAG_OptiX_nobounce(
    camera: EAGCamera,
    sample_renderer: eag_pt_tracer_optix.SampleRenderer,
    surfels: EmissionAwareGaussians,
) -> tuple[EAGTracingResult, float]:
    (
        pixels_hitcounts,
        pixels_alphas,
        pixels_distances,
        pixels_normals,
        pixels_radiances,
        pixels_emissives,
        pixels_albedos,
        time_duration,
    ) = EAG_OptiX_nobounce.apply(
        camera,
        sample_renderer,
        surfels.positions,
        surfels.scales,
        surfels.quaternions,
        surfels.opacities,
        surfels.radiances,
        surfels.emissives,
        surfels.albedos,
    )
    render_results = EAGTracingResult(
        buffer_hitcounts=pixels_hitcounts,
        buffer_alphas=pixels_alphas,
        buffer_distances=pixels_distances,
        buffer_normals=pixels_normals,
        buffer_radiances=pixels_radiances,
        buffer_emissives=pixels_emissives,
        buffer_albedos=pixels_albedos,
    )
    render_results.convertBuffersToImages(
        image_height=camera.image_height,
        image_width=camera.image_width,
    )
    return render_results, time_duration


class EAG_OptiX_nobounce(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx,
        camera: EAGCamera,
        sample_renderer: eag_pt_tracer_optix.SampleRenderer,
        surfels_positions: torch.Tensor,
        surfels_scales: torch.Tensor,
        surfels_quaternions: torch.Tensor,
        surfels_opacities: torch.Tensor,
        surfels_radiances: torch.Tensor,
        surfels_emissives: torch.Tensor,
        surfels_albedos: torch.Tensor,
    ):

        camera_image_height = camera.image_height
        camera_image_width = camera.image_width
        rays = camera.generateRays()
        rays_origins = rays.origins
        rays_directions = rays.directions

        pixels_alphas = torch.zeros(
            (camera.image_height * camera.image_width, 1),
            dtype=torch.float32,
            device="cuda",
        )
        pixels_distances = torch.zeros(
            (camera.image_height * camera.image_width, 1),
            dtype=torch.float32,
            device="cuda",
        )
        pixels_normals = torch.zeros(
            (camera.image_height * camera.image_width, 3),
            dtype=torch.float32,
            device="cuda",
        )
        pixels_radiances = torch.zeros(
            (camera.image_height * camera.image_width, 3),
            dtype=torch.float32,
            device="cuda",
        )

        pixels_emissives = torch.zeros(
            (camera.image_height * camera.image_width, 1),
            dtype=torch.float32,
            device="cuda",
        )
        pixels_albedos = torch.zeros(
            (camera.image_height * camera.image_width, 3),
            dtype=torch.float32,
            device="cuda",
        )
        pixels_hitcounts = torch.zeros(
            (camera.image_height * camera.image_width, 1),
            dtype=torch.int32,
            device="cuda",
        )

        time_start = time.perf_counter()
        sample_renderer.nobounce(
            # [numbers]
            camera_image_height,
            camera_image_width,
            # [input - surfels]
            surfels_positions.contiguous().data_ptr(),
            surfels_scales.contiguous().data_ptr(),
            surfels_quaternions.contiguous().data_ptr(),
            surfels_opacities.contiguous().data_ptr(),
            surfels_radiances.contiguous().data_ptr(),
            surfels_emissives.contiguous().data_ptr(),
            surfels_albedos.contiguous().data_ptr(),
            # [input - rays]
            rays_origins.contiguous().data_ptr(),
            rays_directions.contiguous().data_ptr(),
            # [output - results]
            pixels_hitcounts.contiguous().data_ptr(),
            pixels_alphas.contiguous().data_ptr(),
            pixels_distances.contiguous().data_ptr(),
            pixels_normals.contiguous().data_ptr(),
            pixels_radiances.contiguous().data_ptr(),
            pixels_emissives.contiguous().data_ptr(),
            pixels_albedos.contiguous().data_ptr(),
        )
        time_end = time.perf_counter()
        time_duration = time_end - time_start

        ctx.camera_image_height = camera_image_height
        ctx.camera_image_width = camera_image_width
        ctx.sample_renderer = sample_renderer
        ctx.save_for_backward(
            surfels_positions,
            surfels_scales,
            surfels_quaternions,
            surfels_opacities,
            surfels_radiances,
            surfels_emissives,
            surfels_albedos,
            rays_origins,
            rays_directions,
            pixels_radiances,
            pixels_emissives,
            pixels_alphas,
            pixels_normals,
            pixels_distances,
        )

        return (
            pixels_hitcounts,
            pixels_alphas,
            pixels_distances,
            pixels_normals,
            pixels_radiances,
            pixels_emissives,
            pixels_albedos,
            time_duration,
        )

    @staticmethod
    def backward(
        ctx,
        _useless_d_L_d_pixels_hitcounts,
        d_L_d_pixels_alphas,
        d_L_d_pixels_distances,
        d_L_d_pixels_normals,
        d_L_d_pixels_radiances,
        d_L_d_pixels_emissives,
        d_L_d_pixels_albedos,
        _useless_d_L_d_time_duration,
    ):
        # [get saved from ctx]

        camera_image_height = ctx.camera_image_height
        camera_image_width = ctx.camera_image_width
        sample_renderer = ctx.sample_renderer
        (
            surfels_positions,
            surfels_scales,
            surfels_quaternions,
            surfels_opacities,
            surfels_radiances,
            surfels_emissives,
            surfels_albedos,
            rays_origins,
            rays_directions,
            pixels_radiances,
            pixels_emissives,
            pixels_alphas,
            pixels_normals,
            pixels_distances,
        ) = ctx.saved_tensors

        # [create outputs]

        count_surfels = surfels_positions.shape[0]

        d_L_d_surfels_radiances = torch.zeros(
            (count_surfels, 3),
            dtype=torch.float32,
            device="cuda",
        )
        d_L_d_surfels_emissives = torch.zeros(
            (count_surfels, 1),
            dtype=torch.float32,
            device="cuda",
        )
        d_L_d_surfels_opacities = torch.zeros(
            (count_surfels, 1),
            dtype=torch.float32,
            device="cuda",
        )
        d_L_d_surfels_scales = torch.zeros(
            (count_surfels, 2),
            dtype=torch.float32,
            device="cuda",
        )
        d_L_d_surfels_positions = torch.zeros(
            (count_surfels, 3),
            dtype=torch.float32,
            device="cuda",
        )
        d_L_d_surfels_quaternions = torch.zeros(
            (count_surfels, 4),
            dtype=torch.float32,
            device="cuda",
        )

        # [call backward]

        sample_renderer.nobounce_backward(
            # [numbers]
            camera_image_height,
            camera_image_width,
            # [input - surfels]
            surfels_positions.contiguous().data_ptr(),
            surfels_scales.contiguous().data_ptr(),
            surfels_quaternions.contiguous().data_ptr(),
            surfels_opacities.contiguous().data_ptr(),
            surfels_radiances.contiguous().data_ptr(),
            surfels_emissives.contiguous().data_ptr(),
            surfels_albedos.contiguous().data_ptr(),
            # [input - rays]
            rays_origins.contiguous().data_ptr(),
            rays_directions.contiguous().data_ptr(),
            # [backward - input - forward results]
            pixels_radiances.contiguous().data_ptr(),
            pixels_emissives.contiguous().data_ptr(),
            pixels_alphas.contiguous().data_ptr(),
            pixels_normals.contiguous().data_ptr(),
            pixels_distances.contiguous().data_ptr(),
            # [backward - input - pytorch gradients]
            d_L_d_pixels_radiances.contiguous().data_ptr(),  # (WxH, 3)
            d_L_d_pixels_emissives.contiguous().data_ptr(),  # (WxH, 1)
            d_L_d_pixels_alphas.contiguous().data_ptr(),  # (WxH, 1)
            d_L_d_pixels_normals.contiguous().data_ptr(),  # (WxH, 3)
            d_L_d_pixels_distances.contiguous().data_ptr(),
            # [backward - output]
            d_L_d_surfels_radiances.contiguous().data_ptr(),
            d_L_d_surfels_emissives.contiguous().data_ptr(),
            d_L_d_surfels_opacities.contiguous().data_ptr(),
            d_L_d_surfels_scales.contiguous().data_ptr(),
            d_L_d_surfels_positions.contiguous().data_ptr(),
            d_L_d_surfels_quaternions.contiguous().data_ptr(),
        )

        # [return gradients]

        # ExLog(f"{d_L_d_surfels_positions.sum()=}")

        return (
            None,  # camera: EAGCamera
            None,  # sample_renderer: eag_pt_tracer_optix.SampleRenderer
            # start
            d_L_d_surfels_positions,  # surfels_positions: torch.Tensor
            d_L_d_surfels_scales,  # surfels_scales: torch.Tensor
            d_L_d_surfels_quaternions,  # surfels_quaternions: torch.Tensor
            d_L_d_surfels_opacities,  # surfels_opacities: torch.Tensor
            d_L_d_surfels_radiances,  # surfels_radiances: torch.Tensor
            d_L_d_surfels_emissives,  # surfels_emissives: torch.Tensor
            None,  # surfels_albedos: torch.Tensor
        )


def Differentiable_EAG_OptiX_singlebounce(
    camera: EAGCamera,
    spp: int,
    sample_renderer: eag_pt_tracer_optix.SampleRenderer,
    surfels: EmissionAwareGaussians,
) -> tuple[torch.Tensor, float]:
    pixels_rendering_radiances, duration = EAG_OptiX_singlebounce.apply(
        camera,
        spp,
        sample_renderer,
        surfels.positions,
        surfels.scales,
        surfels.quaternions,
        surfels.opacities,
        surfels.radiances,
        surfels.emissives,
        surfels.albedos,
    )
    return pixels_rendering_radiances, duration


class EAG_OptiX_singlebounce(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx,
        camera: EAGCamera,
        spp: int,
        sample_renderer: eag_pt_tracer_optix.SampleRenderer,
        surfels_positions: torch.Tensor,
        surfels_scales: torch.Tensor,
        surfels_quaternions: torch.Tensor,
        surfels_opacities: torch.Tensor,
        surfels_radiances: torch.Tensor,
        surfels_emissives: torch.Tensor,
        surfels_albedos: torch.Tensor,
    ):

        camera_image_height = camera.image_height
        camera_image_width = camera.image_width
        rays = camera.generateRays()
        rays_origins = rays.origins
        rays_directions = rays.directions

        pixels_albedos = torch.zeros(
            (camera.image_height * camera.image_width, 3),
            dtype=torch.float32,
            device="cuda",
        )
        pixels_rendering_radiances = torch.zeros(
            (camera.image_height * camera.image_width, 3),
            dtype=torch.float32,
            device="cuda",
        )
        d_pixels_rendering_radiances_d_P = torch.zeros(
            (camera.image_height * camera.image_width, 3),
            dtype=torch.float32,
            device="cuda",
        )

        time_start = time.perf_counter()
        sample_renderer.singlebounce(
            # [numbers]
            camera_image_height,
            camera_image_width,
            spp,
            # [input - surfels]
            surfels_positions.contiguous().data_ptr(),
            surfels_scales.contiguous().data_ptr(),
            surfels_quaternions.contiguous().data_ptr(),
            surfels_opacities.contiguous().data_ptr(),
            surfels_radiances.contiguous().data_ptr(),
            surfels_emissives.contiguous().data_ptr(),
            surfels_albedos.contiguous().data_ptr(),
            # [input - rays]
            rays_origins.contiguous().data_ptr(),
            rays_directions.contiguous().data_ptr(),
            # [output - results]
            pixels_albedos.contiguous().data_ptr(),
            pixels_rendering_radiances.contiguous().data_ptr(),
            d_pixels_rendering_radiances_d_P.contiguous().data_ptr(),
        )
        time_end = time.perf_counter()
        time_duration = time_end - time_start

        ctx.camera_image_height = camera_image_height
        ctx.camera_image_width = camera_image_width
        ctx.spp = spp
        ctx.sample_renderer = sample_renderer
        ctx.save_for_backward(
            surfels_positions,
            surfels_scales,
            surfels_quaternions,
            surfels_opacities,
            surfels_radiances,
            surfels_emissives,
            surfels_albedos,
            rays_origins,
            rays_directions,
            pixels_albedos,
            pixels_rendering_radiances,
            d_pixels_rendering_radiances_d_P,
        )

        return (
            pixels_rendering_radiances,
            time_duration,
        )

    @staticmethod
    def backward(
        ctx,
        d_L_d_pixels_rendering_radiances,
        _useless_d_L_d_time_duration,
    ):
        # [get saved from ctx]

        camera_image_height = ctx.camera_image_height
        camera_image_width = ctx.camera_image_width
        spp = ctx.spp
        sample_renderer = ctx.sample_renderer
        (
            surfels_positions,
            surfels_scales,
            surfels_quaternions,
            surfels_opacities,
            surfels_radiances,
            surfels_emissives,
            surfels_albedos,
            rays_origins,
            rays_directions,
            pixels_albedos,
            pixels_rendering_radiances,
            d_pixels_rendering_radiances_d_P,
        ) = ctx.saved_tensors

        # [create outputs]

        count_surfels = surfels_positions.shape[0]

        d_L_d_surfels_albedos = torch.zeros(
            (count_surfels, 3),
            dtype=torch.float32,
            device="cuda",
        )

        # ExLog(f"{(pixels_albedos==0.0).sum()=}")
        # ExLog(f"{(d_pixels_rendering_radiances_d_P.isnan()).sum()=}")

        # [call backward]

        sample_renderer.singlebounce_backward(
            # [numbers]
            camera_image_height,
            camera_image_width,
            # [input - surfels]
            surfels_positions.contiguous().data_ptr(),
            surfels_scales.contiguous().data_ptr(),
            surfels_quaternions.contiguous().data_ptr(),
            surfels_opacities.contiguous().data_ptr(),
            surfels_radiances.contiguous().data_ptr(),
            surfels_emissives.contiguous().data_ptr(),
            surfels_albedos.contiguous().data_ptr(),
            # [input - rays]
            rays_origins.contiguous().data_ptr(),
            rays_directions.contiguous().data_ptr(),
            # [backward - input - forward results]
            pixels_albedos.contiguous().data_ptr(),
            pixels_rendering_radiances.contiguous().data_ptr(),
            d_pixels_rendering_radiances_d_P.contiguous().data_ptr(),
            # [backward - input - pytorch gradients]
            d_L_d_pixels_rendering_radiances.contiguous().data_ptr(),  # (WxH, 3)
            # [backward - output]
            d_L_d_surfels_albedos.contiguous().data_ptr(),
        )

        # [return gradients]

        # ExLog(f"{d_L_d_surfels_positions.sum()=}")

        # if d_L_d_pixels_rendering_radiances.isnan().sum() != 0:
        #     ExLog(f"{d_L_d_pixels_rendering_radiances.isnan().sum()=}")
        #     exit()

        return (
            None,  # camera: EAGCamera
            None,  # spp: int
            None,  # sample_renderer: eag_pt_tracer_optix.SampleRenderer
            # start
            None,  # surfels_positions: torch.Tensor
            None,  # surfels_scales: torch.Tensor
            None,  # surfels_quaternions: torch.Tensor
            None,  # surfels_opacities: torch.Tensor
            None,  # surfels_radiances: torch.Tensor
            None,  # surfels_emissives: torch.Tensor
            d_L_d_surfels_albedos,  # surfels_albedos: torch.Tensor
        )


# [END: add backward]


class EAGNvsDataset:

    def LoadBlenderTransformsJson(
        tracer_config: TracerConfig,
        transforms_json_path: pathlib.Path,
    ) -> list[EAGCamera]:
        ExLog(f"Loading {transforms_json_path}...")
        cameras: list[EAGCamera] = []

        with open(transforms_json_path) as f:
            contents = json.load(f)

        # []

        frames = contents["frames"]
        for i_frame, frame in enumerate(tqdm.tqdm(frames)):

            # [skip train set]

            # 260310: "kitchen" is ambiguous: B- or E-
            # if "kitchen" in str(tracer_config.NVS_DATASET_PATH):
            #     if not frame["file_name"].startswith("23/23_DSC"):
            #         continue
            # elif ("furnishedroom" in str(tracer_config.NVS_DATASET_PATH)) or (
            #     "emptyroom" in str(tracer_config.NVS_DATASET_PATH)
            # ):
            #     if not frame["file_name"].startswith("26/26_DSC"):
            #         continue

            # [remove cabinet in EFT-room]

            # ExLog(f"{int(frame['file_name'][-4:])=}"); exit()
            if (
                "furnishedroom" in str(tracer_config.NVS_DATASET_PATH)
                and int(frame["file_name"][-4:]) < 108
            ):
                continue
            elif (
                "emptyroom" in str(tracer_config.NVS_DATASET_PATH)
                and int(frame["file_name"][-4:]) < 207
            ):
                continue

            # (3 or 4, h, w)
            # [FR]
            # image_radiance_rgb_linear_premultiplied: torch.Tensor = (
            #     UTILITIES_IO.ReadExrImage(
            #         transforms_json_path.parent / f"Image/{frame['file_name']}.exr"
            #     )
            # )
            # [EFT]
            image_radiance_rgb_linear_premultiplied: torch.Tensor = (
                UTILITIES_IO.ReadExrImage(
                    transforms_json_path.parent
                    / f"Radiance-exr/{frame['file_name']}.exr"
                )
            )

            image_width = image_radiance_rgb_linear_premultiplied.shape[2]
            image_height = image_radiance_rgb_linear_premultiplied.shape[1]

            # [transform matrix]

            transform_matrix = torch.tensor(
                frame["transform_matrix"], dtype=torch.float32
            )
            # camera position
            camera_t = transform_matrix[:3, 3]
            # camera rotation (Blender/OpenGL R: X right, -Z forward)
            camera_R_Blender = transform_matrix[:3, :3]
            # Blender R to colmap R
            camera_R_COLMAP = camera_R_Blender.clone()
            camera_R_COLMAP[:, 1:3] *= -1

            Rt = torch.zeros((4, 4), dtype=torch.float32)
            Rt[:3, :3] = camera_R_COLMAP
            Rt[:3, 3] = camera_t
            Rt[3, 3] = 1.0
            view_matrix: torch.Tensor = torch.linalg.inv(Rt)

            # [different camera models: shared through all images]

            if "camera_angle_x" in contents:
                # ExLog(f"Use simple pinhole camera with camera_angle_x.")
                fov_x = contents["camera_angle_x"]
                focal_x = EAGCamera.FovToFocal(fov_x, image_width)
                focal_y = focal_x
                center_x = image_width / 2
                center_y = image_height / 2
            elif (
                "fx" in contents
                and "fy" in contents
                and not "cx" in contents
                and not "cy" in contents
            ):
                # ExLog(f"Use simple pinhole camera with fx and fy.")
                focal_x = contents["fx"]
                focal_y = contents["fy"]
                center_x = image_width / 2
                center_y = image_height / 2
            elif (
                "fx" in contents
                and "fy" in contents
                and "cx" in contents
                and "cy" in contents
            ):
                # ExLog(f"Use pinhole camera with fx,cx and fy,cy.")
                focal_x = contents["fx"]
                center_x = contents["cx"]
                focal_y = contents["fy"]
                center_y = contents["cy"]
            elif "fx" in frame and "fy" in frame and "cx" in frame and "cy" in frame:
                # assign different intrinsic to different camera
                # ExLog(f"Use pinhole camera with fx,cx and fy,cy for each camera.")
                focal_x = frame["fx"]
                center_x = frame["cx"]
                focal_y = frame["fy"]
                center_y = frame["cy"]
            else:
                ExLog(f"Unknow camera model.")
                raise NotImplementedError

            camera = EAGCamera(
                image_width=image_width,
                image_height=image_height,
                focal_x=focal_x,
                center_x=center_x,
                focal_y=focal_y,
                center_y=center_y,
                camera_R=camera_R_Blender,
                camera_t=camera_t,
                view_matrix=view_matrix,
                name=frame["file_name"],
            )

            # [rgb]

            camera.gt_image_radiance_rgb_linear_premultiplied = (
                image_radiance_rgb_linear_premultiplied.cpu()
            )

            # [load image_emissive]

            possible_labeled_emissive_path = (
                transforms_json_path.parent
                / "Emissive-exr"
                / f"{frame['file_name']}.exr"
            )
            if possible_labeled_emissive_path.exists():
                camera.gt_image_emissive = (
                    UTILITIES_IO.ReadExrDepthFromBlender(possible_labeled_emissive_path)
                    .to(dtype=torch.float32)
                    .cpu()
                )
            else:
                camera.gt_image_emissive = None

            # ExLog(f"[DEBUG] {camera.gt_image_emissive.shape=}")

            # [load image_normal and image_depth]

            image_normal = UTILITIES_IO.ReadExrNormalFromBlender(
                transforms_json_path.parent / f"Normal-exr/{frame['file_name']}.exr"
            )
            camera.gt_image_normal = image_normal.cpu()

            if tracer_config.DATASET_IS_SYNTHETIC:

                image_depth = UTILITIES_IO.ReadExrDepthFromBlender(
                    transforms_json_path.parent / f"Depth-exr/{frame['file_name']}.exr"
                )
                camera.gt_image_depth = image_depth.cpu()

                # [load image_albedo]

                image_albedo = UTILITIES_IO.ReadExrDepthFromBlender(
                    transforms_json_path.parent / f"DiffCol/{frame['file_name']}.exr"
                )[:3]
                camera.gt_image_albedo = image_albedo.cpu()

            # [load gt alpha]

            possible_alpha_path = (
                transforms_json_path.parent / "Alpha-exr" / f"{frame['file_name']}.exr"
            )
            if possible_alpha_path.exists():
                image_alpha = UTILITIES_IO.ReadExrDepthFromBlender(possible_alpha_path)
            else:
                image_alpha = torch.ones(
                    (
                        1,
                        image_height,
                        image_width,
                    )
                )

            camera.gt_image_alpha = image_alpha.cpu()

            # [downsample]

            camera.gt_image_radiance_rgb_linear_premultiplied = F.interpolate(
                camera.gt_image_radiance_rgb_linear_premultiplied.unsqueeze(0),
                scale_factor=1.0 / tracer_config.DOWN_SAMPLE_SCALE_WHEN_LOADING_DATA,
                mode="bilinear",
                align_corners=False,
                antialias=True,
            ).squeeze(0)

            width_downsample_scale: float = (
                camera.image_width
                / camera.gt_image_radiance_rgb_linear_premultiplied.shape[2]
            )
            height_downsample_scale: float = (
                camera.image_height
                / camera.gt_image_radiance_rgb_linear_premultiplied.shape[1]
            )
            camera.image_width = (
                camera.gt_image_radiance_rgb_linear_premultiplied.shape[2]
            )
            camera.image_height = (
                camera.gt_image_radiance_rgb_linear_premultiplied.shape[1]
            )
            # ExLog(f"{width_downsample_scale=} {height_downsample_scale=} {camera.image_width=} {camera.image_height=}"); exit()
            camera.focal_x = camera.focal_x / width_downsample_scale
            camera.center_x = (camera.center_x + 0.5) / width_downsample_scale - 0.5
            camera.focal_y = camera.focal_y / height_downsample_scale
            camera.center_y = (camera.center_y + 0.5) / height_downsample_scale - 0.5

            if camera.gt_image_emissive != None:
                camera.gt_image_emissive = F.interpolate(
                    camera.gt_image_emissive.unsqueeze(0),
                    scale_factor=1.0
                    / tracer_config.DOWN_SAMPLE_SCALE_WHEN_LOADING_DATA,
                    mode="bilinear",
                    align_corners=False,
                    antialias=True,
                ).squeeze(0)
            camera.gt_image_normal = F.interpolate(
                camera.gt_image_normal.unsqueeze(0),
                scale_factor=1.0 / tracer_config.DOWN_SAMPLE_SCALE_WHEN_LOADING_DATA,
                mode="bilinear",
                align_corners=False,
                antialias=True,
            ).squeeze(0)
            camera.gt_image_alpha = F.interpolate(
                camera.gt_image_alpha.unsqueeze(0),
                scale_factor=1.0 / tracer_config.DOWN_SAMPLE_SCALE_WHEN_LOADING_DATA,
                mode="bilinear",
                align_corners=False,
                antialias=True,
            ).squeeze(0)

            # [return]

            cameras.append(camera)

        return cameras

    def LoadBlenderTransformsJsonTrainAndTest(
        transforms_train_json_path: pathlib.Path,
        transforms_test_json_path: pathlib.Path,
    ) -> EAGNvsDataset:
        return EAGNvsDataset(
            train_set_cameras=EAGNvsDataset.LoadBlenderTransformsJson(
                transforms_train_json_path
            ),
            test_set_cameras=EAGNvsDataset.LoadBlenderTransformsJson(
                transforms_test_json_path
            ),
        )

    def LoadBlenderTransformsSingle(
        tracer_config: TracerConfig,
        transforms_json_path: pathlib.Path,
    ) -> EAGNvsDataset:
        all_cameras: list[EAGCamera] = EAGNvsDataset.LoadBlenderTransformsJson(
            tracer_config, transforms_json_path
        )

        # ExLog(f"{all_cameras=}")

        if "/" in all_cameras[0].name:  # EFT dataset
            train_cam_infos = [
                cam
                for cam in all_cameras
                if (
                    not cam.name.startswith("10/10_DSC")
                    and not cam.name.startswith("11/11_DSC")
                    and not cam.name.startswith("12/12_DSC")
                )
            ]
            test_cam_infos = [
                cam
                for cam in all_cameras
                if (
                    cam.name.startswith("26/26_DSC")
                    if not "kitchen" in str(tracer_config.NVS_DATASET_PATH)
                    else cam.name.startswith("23/23_DSC")
                )
            ]
            print(f"[DEBUG] {len(train_cam_infos)=} {len(test_cam_infos)=}")
            return EAGNvsDataset(
                train_set_cameras=train_cam_infos,
                test_set_cameras=test_cam_infos,
            )
        if "SelfCaptured" in str(tracer_config.NVS_DATASET_PATH):
            return EAGNvsDataset(
                train_set_cameras=all_cameras,
                test_set_cameras=all_cameras,
            )
        else:  # Blender and FR dataset
            return EAGNvsDataset(
                train_set_cameras=[
                    camera for i, camera in enumerate(all_cameras) if i % 8 != 0
                ],
                test_set_cameras=[
                    camera for i, camera in enumerate(all_cameras) if i % 8 == 0
                ],
            )
            print(f"[DEBUG] {len(train_cam_infos)=} {len(test_cam_infos)=}")

    def __init__(
        self,
        train_set_cameras: list[EAGCamera],
        test_set_cameras: list[EAGCamera],
    ) -> None:
        self.train_set_cameras: list[EAGCamera] = train_set_cameras
        self.test_set_cameras: list[EAGCamera] = test_set_cameras


class EAGCamera:

    def FocalToFov(focal: float, pixels: int) -> float:
        return 2 * math.atan(pixels / (2 * focal))

    def FovToFocal(fov: float, pixels: int) -> float:
        return pixels / (2 * math.tan(fov / 2))

    def __init__(
        self,
        image_width: int,
        image_height: int,
        focal_x: float,
        center_x: float,
        focal_y: float,
        center_y: float,
        camera_R: torch.tensor,
        camera_t: torch.tensor,
        view_matrix: torch.tensor,
        name: str = "None",
    ) -> None:
        self.name = name

        self.image_width: int = image_width
        self.image_height: int = image_height
        self.focal_x: int = focal_x
        self.center_x: int = center_x
        self.focal_y: int = focal_y
        self.center_y: int = center_y

        self.camera_R: torch.tensor = camera_R
        self.camera_t: torch.tensor = camera_t

        self.view_matrix: torch.tensor = view_matrix

        self.fov_x = EAGCamera.FocalToFov(focal=self.focal_x, pixels=self.image_width)
        self.fov_y = EAGCamera.FocalToFov(focal=self.focal_y, pixels=self.image_height)

        # save gt data inside the camera

        self.gt_image_radiance_rgb_linear_premultiplied: torch.Tensor | None = None
        self.path_tracing_radiance_rgb_linear: torch.Tensor | None = None
        self.gt_image_alpha: torch.Tensor | None = None

        self.gt_image_emissive: torch.Tensor | None = None

        self.gt_image_albedo: torch.Tensor | None = None

        self.gt_image_depth: torch.Tensor | None = None
        self.gt_image_normal: torch.Tensor | None = None

    def generateRays(
        self,
    ) -> EAGRays:
        count_rays = self.image_height * self.image_width

        # [ray_indices]

        # (hxw,)
        ray_indices = torch.arange(count_rays, dtype=torch.int32)

        # [rays_origins]

        # (hxw, 3)
        rays_origins = torch.tile(input=self.camera_t, dims=(count_rays, 1))

        # [rays_directions]

        # (hxw, 3)
        rays_directions = torch.ones((count_rays, 3), dtype=torch.float32)
        pixel_columns = (ray_indices % self.image_width)[:, None]
        rays_directions[:, [0]] = (pixel_columns + 0.5 - self.center_x) / self.focal_x
        pixel_rows = (ray_indices // self.image_width)[:, None]
        rays_directions[:, [1]] = (pixel_rows + 0.5 - self.center_y) / self.focal_y

        rays_directions = rays_directions @ self.view_matrix[:3, :3]
        rays_directions = torch.nn.functional.normalize(rays_directions, dim=-1)

        return EAGRays(
            count=count_rays,
            ray_indices=ray_indices,
            origins=rays_origins,
            directions=rays_directions,
            facetos=(torch.tensor([[0.0, 0.0, 1.0]]) @ self.view_matrix[:3, :3]).repeat(
                count_rays, 1
            ),
        )


class EAGRays:

    def __init__(
        self,
        count: int,
        # (N,) int32
        ray_indices: torch.Tensor,
        # (N, 3) float32
        origins: torch.Tensor,
        # (N, 3) float32
        directions: torch.Tensor,
        # (N, 3) float32
        facetos: torch.Tensor,
    ) -> None:
        self.count: int = count
        self.ray_indices: torch.Tensor = ray_indices
        self.origins: torch.Tensor = origins
        self.directions: torch.Tensor = directions
        self.facetos: torch.Tensor = facetos

    def filter(
        self,
        # (self.count,)
        mask: torch.Tensor,
    ) -> EAGRays:
        # TODO index mask or bool mask?
        assert mask.shape[0] == self.count
        return EAGRays(
            count=mask.sum().item(),
            ray_indices=self.ray_indices[mask],
            origins=self.origins[mask],
            directions=self.directions[mask],
        )


class EAGTracingResult:

    # pixel_color to color_pixel
    def pc2cp(
        buffer: torch.Tensor,
        image_height: int,
        image_width: int,
    ) -> torch.Tensor:
        assert buffer.shape[0] == image_height * image_width
        data_length = buffer.shape[1]
        return einops.rearrange(buffer, "p c -> c p").reshape(
            (data_length, image_height, image_width)
        )

    # this function should be differentiable
    def DeriveDepthFromDistanceGivenCamera(
        buffer_distances: torch.Tensor,
        camera: EAGCamera,
    ) -> torch.Tensor:

        camera_rays: EAGRays = camera.generateRays()

        image_depth = EAGTracingResult.pc2cp(
            buffer_distances
            * (camera_rays.directions * camera_rays.facetos).sum(dim=1, keepdim=True),
            camera.image_height,
            camera.image_width,
        )
        return image_depth

    # this function should be differentiable
    def DeriveNormalFromDistanceGivenCamera(
        buffer_distances: torch.Tensor,
        camera: EAGCamera,
    ) -> torch.Tensor:

        camera_rays: EAGRays = camera.generateRays()

        buffer_3d_points = (
            camera_rays.origins + camera_rays.directions * buffer_distances
        )
        image_points = EAGTracingResult.pc2cp(
            buffer_3d_points, camera.image_height, camera.image_width
        )

        dx = image_points[:, 2:, 1:-1] - image_points[:, :-2, 1:-1]
        dy = image_points[:, 1:-1, 2:] - image_points[:, 1:-1, :-2]
        normal_map = torch.nn.functional.normalize(torch.cross(dx, dy, dim=0), dim=0)

        depth_normal = torch.zeros((3, camera.image_height, camera.image_width))
        depth_normal[:, 1:-1, 1:-1] = normal_map

        return depth_normal

    def __init__(
        self,
        buffer_hitcounts: torch.Tensor | None = None,
        buffer_alphas: torch.Tensor | None = None,
        #
        buffer_distances: torch.Tensor | None = None,
        buffer_normals: torch.Tensor | None = None,
        #
        buffer_radiances: torch.Tensor | None = None,
        buffer_emissives: torch.Tensor | None = None,
        buffer_albedos: torch.Tensor | None = None,
    ) -> None:

        self.buffer_hitcounts: torch.Tensor | None = buffer_hitcounts
        self.buffer_alphas: torch.Tensor | None = buffer_alphas

        # avoid distance==0.0 and alpha==0.0
        if False:
            # this also works
            distance = buffer_distances / buffer_alphas
            self.buffer_distances: torch.Tensor | None = torch.nan_to_num(
                distance, 0.0, 0.0, 0.0
            )
        if True:
            self.buffer_distances: torch.Tensor | None = torch.zeros_like(
                buffer_distances
            )
            mask = buffer_hitcounts != 0
            self.buffer_distances[mask] = buffer_distances[mask] / buffer_alphas[mask]

        self.buffer_normals: torch.Tensor | None = torch.nn.functional.normalize(
            buffer_normals, dim=1
        )

        self.buffer_radiances: torch.Tensor | None = buffer_radiances
        self.buffer_emissives: torch.Tensor | None = buffer_emissives
        self.buffer_albedos: torch.Tensor | None = buffer_albedos

    def convertBuffersToImages(
        self, image_height: int, image_width: int
    ) -> torch.Tensor:

        if self.buffer_hitcounts != None:
            self.image_hitcount = EAGTracingResult.pc2cp(
                buffer=self.buffer_hitcounts,
                image_height=image_height,
                image_width=image_width,
            )
        if self.buffer_alphas != None:
            self.image_alpha = EAGTracingResult.pc2cp(
                buffer=self.buffer_alphas,
                image_height=image_height,
                image_width=image_width,
            )
        #
        if self.buffer_distances != None:
            self.image_distance = EAGTracingResult.pc2cp(
                buffer=self.buffer_distances,
                image_height=image_height,
                image_width=image_width,
            )
        if self.buffer_normals != None:
            self.image_normal = EAGTracingResult.pc2cp(
                buffer=self.buffer_normals,
                image_height=image_height,
                image_width=image_width,
            )
        #
        if self.buffer_radiances != None:
            self.image_radiance = EAGTracingResult.pc2cp(
                buffer=self.buffer_radiances,
                image_height=image_height,
                image_width=image_width,
            )
        if self.buffer_emissives != None:
            self.image_emissive = EAGTracingResult.pc2cp(
                buffer=self.buffer_emissives,
                image_height=image_height,
                image_width=image_width,
            )
        if self.buffer_albedos != None:
            self.image_albedo = EAGTracingResult.pc2cp(
                buffer=self.buffer_albedos,
                image_height=image_height,
                image_width=image_width,
            )


class EmissionAwareGaussians:

    def Normalize(x: torch.Tensor) -> torch.Tensor:
        norm = x.pow(2).sum(dim=1).sqrt()[:, None]
        return x / norm

    def LoadPly(path: pathlib.Path) -> EmissionAwareGaussians:
        points: plyfile.PlyElement = plyfile.PlyData.read(path)["vertex"]
        ExLog(f"Read {points.count} points from {path}.")

        gsply_positions: np.ndarray = np.column_stack(
            (
                points["x"],
                points["y"],
                points["z"],
            )
        )
        gsply_scales: np.ndarray = np.column_stack(
            (
                points["scales_0"],
                points["scales_1"],
            )
        )
        gsply_quaternions: np.ndarray = np.column_stack(
            (
                points["quaternions_0"],
                points["quaternions_1"],
                points["quaternions_2"],
                points["quaternions_3"],
            )
        )
        gsply_opacities: np.ndarray = np.column_stack((points["opacities"],)).astype(
            np.float32
        )
        gsply_sh0s: np.ndarray = np.column_stack(
            (
                points["radiances_0"],
                points["radiances_1"],
                points["radiances_2"],
            )
        )

        gsply_emissives: np.ndarray = np.column_stack((points["emissives"],))

        gsply_albedos: np.ndarray = np.column_stack(
            (
                points["albedos_0"],
                points["albedos_1"],
                points["albedos_2"],
            )
        )

        return EmissionAwareGaussians(
            count=points.count,
            positions=torch.tensor(gsply_positions, dtype=torch.float32),
            scales=torch.tensor(gsply_scales, dtype=torch.float32),
            quaternions=torch.tensor(gsply_quaternions, dtype=torch.float32),
            opacities=torch.tensor(gsply_opacities, dtype=torch.float32),
            radiances=torch.tensor(gsply_sh0s, dtype=torch.float32),
            emissives=torch.tensor(gsply_emissives, dtype=torch.float32),
            albedos=torch.tensor(gsply_albedos, dtype=torch.float32),
        )

    def GetGaussiansTangentAndNormalVectors(
        count: int,
        # (N_gaussians, 4)
        quaternions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        quaternions_norms = torch.sqrt(
            quaternions[:, 0] * quaternions[:, 0]
            + quaternions[:, 1] * quaternions[:, 1]
            + quaternions[:, 2] * quaternions[:, 2]
            + quaternions[:, 3] * quaternions[:, 3]
        )[:, None]
        normalized_quaternions = quaternions / quaternions_norms

        r = normalized_quaternions[:, 0]
        x = normalized_quaternions[:, 1]
        y = normalized_quaternions[:, 2]
        z = normalized_quaternions[:, 3]

        R = torch.zeros((count, 3, 3), dtype=torch.float32)

        R[:, 0, 0] = 1 - 2 * (y * y + z * z)
        R[:, 1, 0] = 2 * (x * y + r * z)
        R[:, 2, 0] = 2 * (x * z - r * y)

        R[:, 0, 1] = 2 * (x * y - r * z)
        R[:, 1, 1] = 1 - 2 * (x * x + z * z)
        R[:, 2, 1] = 2 * (y * z + r * x)

        R[:, 0, 2] = 2 * (x * z + r * y)
        R[:, 1, 2] = 2 * (y * z - r * x)
        R[:, 2, 2] = 1 - 2 * (x * x + y * y)

        return R[:, :, 0], R[:, :, 1], R[:, :, 2]  # (N_gaussians, 3)

    def GetGaussiansRotationMatrices(
        count: int,
        # (N_gaussians, 4)
        quaternions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        quaternions_norms = torch.sqrt(
            quaternions[:, 0] * quaternions[:, 0]
            + quaternions[:, 1] * quaternions[:, 1]
            + quaternions[:, 2] * quaternions[:, 2]
            + quaternions[:, 3] * quaternions[:, 3]
        )[:, None]
        normalized_quaternions = quaternions / quaternions_norms

        r = normalized_quaternions[:, 0]
        x = normalized_quaternions[:, 1]
        y = normalized_quaternions[:, 2]
        z = normalized_quaternions[:, 3]

        R = torch.zeros((count, 3, 3), dtype=torch.float32)
        R[:, 0, 0] = 1 - 2 * (y * y + z * z)
        R[:, 0, 1] = 2 * (x * y - r * z)
        R[:, 0, 2] = 2 * (x * z + r * y)
        R[:, 1, 0] = 2 * (x * y + r * z)
        R[:, 1, 1] = 1 - 2 * (x * x + z * z)
        R[:, 1, 2] = 2 * (y * z - r * x)
        R[:, 2, 0] = 2 * (x * z - r * y)
        R[:, 2, 1] = 2 * (y * z + r * x)
        R[:, 2, 2] = 1 - 2 * (x * x + y * y)

        return R

    def InitializeGaussiansFromPointCloudFile(
        ply_path: pathlib.Path,
    ) -> EmissionAwareGaussians:
        points: plyfile.PlyElement = plyfile.PlyData.read(ply_path)["vertex"]
        ExLog(f"Read {points.count} points from {ply_path}.")

        pc_positions: np.ndarray = np.column_stack(
            (
                points["x"],
                points["y"],
                points["z"],
            )
        )

        pc_rgbs: np.ndarray = (
            np.column_stack((points["red"], points["green"], points["blue"])).astype(
                np.float32
            )
            / 255.0
        )

        random_quaternions = torch.rand((points.count, 4), dtype=torch.float32)
        random_quaternions = random_quaternions / (
            (random_quaternions.square()).sum(dim=1, keepdim=True).sqrt()
        )

        return EmissionAwareGaussians(
            count=points.count,
            positions=torch.tensor(pc_positions),
            scales=torch.ones((points.count, 2), dtype=torch.float32) * 0.01,
            quaternions=random_quaternions,
            opacities=torch.ones((points.count, 1), dtype=torch.float32) * 0.1,
            radiances=torch.tensor(pc_rgbs),
            emissives=torch.ones((points.count, 1), dtype=torch.float32) * 0.1,
            albedos=torch.ones((points.count, 3), dtype=torch.float32) * 0.25,
        )

    def __init__(
        self,
        count: int,
        positions: torch.Tensor,
        scales: torch.Tensor,
        quaternions: torch.Tensor,
        opacities: torch.Tensor,
        radiances: torch.Tensor,
        emissives: torch.Tensor,
        albedos: torch.Tensor,
    ) -> None:
        # count of 2D Gaussians, row count of properties
        self.count: int = count

        # (n, 3)
        self.positions: torch.Tensor = positions
        # (n, 2)
        self.scales: torch.Tensor = scales
        # (n, 4)
        self.quaternions: torch.Tensor = quaternions

        # (n, 1)
        self.opacities: torch.Tensor = opacities
        # (n, 3)
        # linear rgb
        self.radiances: torch.Tensor = radiances

        # (n, 1)
        # non-emissive (0.0) or emissive (1.0)
        self.emissives: torch.Tensor = emissives

        # (n, 3)
        self.albedos: torch.Tensor = albedos

    def filter(self, mask_bool: torch.Tensor) -> EmissionAwareGaussians:
        return EmissionAwareGaussians(
            count=mask_bool.sum().item(),
            positions=self.positions[mask_bool].clone(),
            scales=self.scales[mask_bool].clone(),
            quaternions=self.quaternions[mask_bool].clone(),
            opacities=self.opacities[mask_bool].clone(),
            radiances=self.radiances[mask_bool].clone(),
            emissives=self.emissives[mask_bool].clone(),
            albedos=self.albedos[mask_bool].clone(),
        )

    def merge(self, extra_gaussians: EmissionAwareGaussians) -> EmissionAwareGaussians:
        return EmissionAwareGaussians(
            count=self.count + extra_gaussians.count,
            positions=torch.cat([self.positions, extra_gaussians.positions]),
            scales=torch.cat([self.scales, extra_gaussians.scales]),
            quaternions=torch.cat([self.quaternions, extra_gaussians.quaternions]),
            opacities=torch.cat([self.opacities, extra_gaussians.opacities]),
            radiances=torch.cat([self.radiances, extra_gaussians.radiances]),
            emissives=torch.cat([self.emissives, extra_gaussians.emissives]),
            albedos=torch.cat([self.albedos, extra_gaussians.albedos]),
        )

    @torch.no_grad()
    def get_triangles_vertices_and_indices_for_optix_build_acceleration_structure(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        SIGMA_VALUE = 3.0

        # (4N, 3)
        gaussians_vertices = torch.tensor(
            [[-1.0, -1.0, 0.0], [1.0, -1.0, 0.0], [-1.0, 1.0, 0.0], [1.0, 1.0, 0.0]]
        ).repeat(self.count, 1)
        gaussians_vertices[:, 0:2] = (
            gaussians_vertices[:, 0:2]
            * self.scales.repeat_interleave(4, dim=0)
            * SIGMA_VALUE
        )

        # (4N, 3, 3)
        gaussians_Rs: torch.Tensor = (
            EmissionAwareGaussians.GetGaussiansRotationMatrices(
                count=self.count, quaternions=self.quaternions
            ).repeat_interleave(4, dim=0)
        )

        # (4N, 1, 3)
        surfels_rotated_vertices = gaussians_vertices[
            :, None, :
        ] @ gaussians_Rs.transpose(1, 2)
        # (4N, 3)
        surfels_rotated_vertices = surfels_rotated_vertices[:, 0, :]
        # (4N, 3)
        surfels_rotated_and_moved_vertices = (
            surfels_rotated_vertices + self.positions.repeat_interleave(4, dim=0)
        )

        # (2, 3)
        surfels_indices_offset = torch.tensor([[0, 1, 2], [3, 2, 1]])
        # (2N, 3)
        surfels_indices_offsets = surfels_indices_offset.repeat(self.count, 1)
        # (N, 1)
        surfels_indices_base = torch.arange(self.count)[:, None] * 4
        # (2N, 1)
        surfels_indices_bases = surfels_indices_base.repeat_interleave(2, dim=0)
        # (2N, 3)
        surfels_indices = surfels_indices_bases + surfels_indices_offsets

        return surfels_rotated_and_moved_vertices.to(
            dtype=torch.float32
        ), surfels_indices.to(dtype=torch.int32)

    # Save activated properties.
    def savePly(self, path: pathlib.Path) -> None:
        ply_points = np.concatenate(
            [
                self.positions.cpu().numpy(),
                self.scales.cpu().numpy(),
                self.quaternions.cpu().numpy(),
                self.opacities.cpu().numpy(),
                self.radiances.cpu().numpy(),
                self.emissives.cpu().numpy(),
                self.albedos.cpu().numpy(),
            ],
            axis=1,
        )
        ply_properties = (
            [
                ("x", "f4"),
                ("y", "f4"),
                ("z", "f4"),
            ]
            + [
                ("scales_0", "f4"),
                ("scales_1", "f4"),
            ]
            + [
                ("quaternions_0", "f4"),
                ("quaternions_1", "f4"),
                ("quaternions_2", "f4"),
                ("quaternions_3", "f4"),
            ]
            + [("opacities", "f4")]
            + [
                ("radiances_0", "f4"),
                ("radiances_1", "f4"),
                ("radiances_2", "f4"),
            ]
            + [("emissives", "f4")]
            + [
                ("albedos_0", "f4"),
                ("albedos_1", "f4"),
                ("albedos_2", "f4"),
            ]
        )

        UTILITIES_IO.SavePlyUsingPlyfilePackage(
            path=path,
            points=ply_points,
            properties=ply_properties,
        )
        ExLog(f"Saved {self.count} points to {path}.")

    def saveColoredPointCloud(self, path: pathlib.Path) -> None:
        ply_points = np.concatenate(
            [
                self.positions.cpu().numpy(),
                (UTILITIES_COLOUR.LinearToSrgb(self.radiances).clip(0.0, 1.0) * 255.0)
                .cpu()
                .numpy()
                .astype(np.uint8),
            ],
            axis=1,
        )
        ply_properties = [
            ("x", "f4"),
            ("y", "f4"),
            ("z", "f4"),
        ] + [
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
        ]
        UTILITIES_IO.SavePlyUsingPlyfilePackage(
            path=path,
            points=ply_points,
            properties=ply_properties,
        )

    def saveColoredPointCloudEmissivesWithThreshold(
        self,
        path: pathlib.Path,
        threshold: float = 0.5,
    ) -> None:
        # colors = torch.zeros_like(self.positions)
        colors = UTILITIES_COLOUR.LinearToSrgb(self.radiances).clip(0.0, 1.0)
        colors[(self.emissives > threshold)[:, 0], 1] = 1.0  # green
        colors[(self.emissives > threshold)[:, 0], 0] = 0.0  # green
        colors[(self.emissives > threshold)[:, 0], 2] = 0.0  # green

        ply_points = np.concatenate(
            [
                self.positions.cpu().numpy(),
                (colors * 255.0).cpu().numpy().astype(np.uint8),
            ],
            axis=1,
        )
        ply_properties = [
            ("x", "f4"),
            ("y", "f4"),
            ("z", "f4"),
        ] + [
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
        ]
        UTILITIES_IO.SavePlyUsingPlyfilePackage(
            path=path,
            points=ply_points,
            properties=ply_properties,
        )

    @torch.no_grad()
    def saveColoredBoundingRectangles(
        self, path: pathlib.Path, scale_multiple: float = 1.0
    ) -> None:

        triangles_vertices, triangles_ids = (
            self.get_triangles_vertices_and_indices_for_optix_build_acceleration_structure()
        )

        triangles_vertex_colors = torch.repeat_interleave(
            UTILITIES_COLOUR.LinearToSrgb(self.radiances).clip(0.0, 1.0)
            * self.opacities,
            # UTILITIES_COLOUR.LinearToSrgb(self.radiances).clip(0.0, 1.0),
            4,
            dim=0,
        )

        ply_points = np.concatenate(
            [
                triangles_vertices.cpu().numpy(),
                (triangles_vertex_colors * 255.0).cpu().numpy().astype(np.uint8),
            ],
            axis=1,
        )
        ply_properties = [
            ("x", "f4"),
            ("y", "f4"),
            ("z", "f4"),
        ] + [
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
        ]

        vertices = plyfile.PlyElement.describe(
            np.array(list(map(tuple, ply_points)), dtype=ply_properties),
            "vertex",
        )
        faces = plyfile.PlyElement.describe(
            np.array(
                [(face,) for face in triangles_ids.cpu().numpy()],
                dtype=[("vertex_indices", "i4", (3,))],
            ),
            "face",
        )
        plyfile.PlyData([vertices, faces]).write(path)

    def saveTsdfMeshFromDepthAndRgb(
        self,
        tracer_config: TracerConfig,
        train_set_cameras: list[EAGCamera],
        folder: pathlib.Path,
    ) -> None:
        class Frame:
            def __init__(
                self,
                width,
                height,
                fx,
                fy,
                cx,
                cy,
                world_view_transform,
                rgb,
                depth,
            ) -> None:

                self.width = width
                self.height = height
                self.fx = fx
                self.fy = fy
                self.cx = cx
                self.cy = cy

                self.world_view_transform: np.ndarray = world_view_transform

                intrinsic = o3d.camera.PinholeCameraIntrinsic(
                    width=width,
                    height=height,
                    cx=cx,
                    cy=cy,
                    fx=fx,
                    fy=fy,
                )
                extrinsic = np.asarray((world_view_transform.T).cpu().numpy())
                camera = o3d.camera.PinholeCameraParameters()
                camera.extrinsic = extrinsic
                camera.intrinsic = intrinsic
                self.o3d_camera = camera

                self.rgb: torch.Tensor = rgb
                self.depth: torch.Tensor = depth

        def post_process_mesh(mesh, cluster_to_keep=1000):
            """
            Post-process a mesh to filter out floaters and disconnected parts
            """

            print(
                "post processing the mesh to have {} clusterscluster_to_kep".format(
                    cluster_to_keep
                )
            )
            mesh_0 = copy.deepcopy(mesh)
            with o3d.utility.VerbosityContextManager(
                o3d.utility.VerbosityLevel.Debug
            ) as cm:
                triangle_clusters, cluster_n_triangles, cluster_area = (
                    mesh_0.cluster_connected_triangles()
                )

            triangle_clusters = np.asarray(triangle_clusters)
            cluster_n_triangles = np.asarray(cluster_n_triangles)
            cluster_area = np.asarray(cluster_area)
            if cluster_to_keep < len(cluster_n_triangles):
                n_cluster = np.sort(cluster_n_triangles.copy())[-cluster_to_keep]
            else:
                n_cluster = 50
            n_cluster = max(n_cluster, 50)  # filter meshes smaller than 50
            triangles_to_remove = cluster_n_triangles[triangle_clusters] < n_cluster
            mesh_0.remove_triangles_by_mask(triangles_to_remove)
            mesh_0.remove_unreferenced_vertices()
            mesh_0.remove_degenerate_triangles()
            print("num vertices raw {}".format(len(mesh.vertices)))
            print("num vertices post {}".format(len(mesh_0.vertices)))
            return mesh_0

        # [START: optix backend - build scene acceleration structure]

        sample_renderer = eag_pt_tracer_optix.SampleRenderer()
        triangles_vertices, triangles_indices = (
            self.get_triangles_vertices_and_indices_for_optix_build_acceleration_structure()
        )
        sample_renderer.buildAccel(
            triangles_vertices.shape[0],
            triangles_vertices.contiguous().data_ptr(),
            triangles_indices.shape[0],
            triangles_indices.contiguous().data_ptr(),
        )

        # [END: optix backend - build scene acceleration structure]

        depth_trunc = tracer_config.TSDF_MESH_EXPORT_DEPTH_TRUNC
        voxel_size = depth_trunc / tracer_config.TSDF_MESH_EXPORT_MESH_RESOLUTION
        sdf_trunc = 5.0 * voxel_size

        print("Running tsdf volume integration ...")
        print(f"voxel_size: {voxel_size}")
        print(f"sdf_trunc: {sdf_trunc}")
        print(f"depth_truc: {depth_trunc}")

        volume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=voxel_size,
            sdf_trunc=sdf_trunc,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
        )
        with torch.no_grad():
            for i_camera, camera in enumerate(train_set_cameras):

                render_results, _ = Differentiable_EAG_OptiX_nobounce(
                    camera,
                    sample_renderer,
                    self,
                )

                rgb = UTILITIES_COLOUR.LinearToSrgb(render_results.image_radiance).clip(
                    0.0, 1.0
                )
                depth = EAGTracingResult.DeriveDepthFromDistanceGivenCamera(
                    buffer_distances=render_results.buffer_distances,
                    camera=camera,
                )

                current_frame = Frame(
                    width=camera.image_width,
                    height=camera.image_height,
                    fx=camera.focal_x,
                    fy=camera.focal_y,
                    cx=camera.center_x,
                    cy=camera.center_y,
                    world_view_transform=camera.view_matrix.T,
                    rgb=rgb,
                    depth=depth,
                )

                rgb = current_frame.rgb
                depth = current_frame.depth

                # make open3d rgbd
                rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                    o3d.geometry.Image(
                        np.asarray(
                            np.clip(rgb.permute(1, 2, 0).cpu().numpy(), 0.0, 1.0) * 255,
                            order="C",
                            dtype=np.uint8,
                        )
                    ),
                    o3d.geometry.Image(
                        np.asarray(depth.permute(1, 2, 0).cpu().numpy(), order="C")
                    ),
                    depth_trunc=depth_trunc,
                    convert_rgb_to_intensity=False,
                    depth_scale=1.0,
                )

                volume.integrate(
                    rgbd,
                    intrinsic=current_frame.o3d_camera.intrinsic,
                    extrinsic=current_frame.o3d_camera.extrinsic,
                )

        mesh = volume.extract_triangle_mesh()

        mesh_path = folder / "tsdf-mesh.ply"
        o3d.io.write_triangle_mesh(mesh_path, mesh)
        print(f"[LOG] mesh saved at {mesh_path}")

        mesh_post = post_process_mesh(
            mesh,
            cluster_to_keep=tracer_config.TSDF_MESH_EXPORT_NUM_CLUSTERS,
        )

        mesh_post_path = folder / "tsdf-mesh-post.ply"
        o3d.io.write_triangle_mesh(mesh_post_path, mesh_post)
        print(f"[LOG] mesh post processed saved at {mesh_post_path}")

        mesh_post_path = folder / "scene.obj"
        o3d.io.write_triangle_mesh(mesh_post_path, mesh_post)
        print(f"[LOG] mesh post processed saved at {mesh_post_path}")

    @torch.no_grad()
    def saveNoBounceResultsOnCameras(
        self,
        tracer_config: TracerConfig,
        cameras: list[EAGCamera],
        folder_path: pathlib.Path,
        bounding_rectanges_scale_multiple: float = 3.0,
        is_to_save_groundtruths: bool = False,
    ) -> None:

        # [prepare records]

        record_durations = []
        record_psnrs_radiance_linear = []
        record_psnrs_radiance_pq = []
        record_psnrs_radiance_srgbclip = []
        record_lpipss_radiance_srgbclip = []
        record_flips_radiance_linear = []
        record_psnrs_emissive_linear = []
        if tracer_config.DATASET_IS_SYNTHETIC:
            record_psnrs_albedo_linear = []
            record_psnrs_albedo_srgbclip = []
        record_psnrs_alpha_linear = []

        # [optix backend - build scene acceleration structure]

        sample_renderer = eag_pt_tracer_optix.SampleRenderer()
        triangles_vertices, triangles_indices = (
            self.get_triangles_vertices_and_indices_for_optix_build_acceleration_structure()
        )
        sample_renderer.buildAccel(
            triangles_vertices.shape[0],
            triangles_vertices.contiguous().data_ptr(),
            triangles_indices.shape[0],
            triangles_indices.contiguous().data_ptr(),
        )

        for i_camera, camera in enumerate(cameras):

            # [load gt from cpu memory to gpu memory]

            gt_image_radiance = camera.gt_image_radiance_rgb_linear_premultiplied.cuda()
            gt_image_radiance_pq = UTILITIES_COLOUR.LinearToPq(gt_image_radiance)
            gt_image_radiance_rgb_srgbclip_premultiplied: torch.Tensor = (
                UTILITIES_COLOUR.LinearToSrgb(gt_image_radiance).clip(min=0.0, max=1.0)
            )

            gt_image_alpha = camera.gt_image_alpha.cuda()

            if camera.gt_image_emissive != None:
                gt_image_emissive = camera.gt_image_emissive.cuda()
            else:
                gt_image_emissive = None

            if tracer_config.DATASET_IS_SYNTHETIC:
                gt_image_albedo = camera.gt_image_albedo.cuda()

            # [save gt: rgb/a/rgba emissives depth/normal]

            if is_to_save_groundtruths:
                UTILITIES_IO.SaveExrImage(
                    gt_image_radiance,
                    folder_path / f"gt-Radiance-exr/camera{i_camera}-Radiance-gt.exr",
                )

                UTILITIES_IO.SaveExrImage(
                    gt_image_alpha,
                    folder_path / f"gt-Alpha-exr/camera{i_camera}-Alpha-gt.exr",
                )

                if gt_image_emissive != None:
                    UTILITIES_IO.SaveExrImage(
                        gt_image_emissive,
                        folder_path
                        / f"gt-Emissive-exr/camera{i_camera}-Emissive-gt.exr",
                    )
                    UTILITIES_IO.SaveExrImage(
                        gt_image_emissive * torch.tensor([0.0, 1.0, 0.0])[:, None, None]
                        + (1.0 - gt_image_emissive) * gt_image_radiance,
                        folder_path
                        / f"gt-Emissive-exr/camera{i_camera}-Emissive-gt-w-radiances.exr",
                    )

                UTILITIES_IO.SaveExrImage(
                    camera.gt_image_normal,
                    folder_path / f"gt-Normal-exr/camera{i_camera}-Normal-gt.exr",
                )
                UTILITIES_IO.SaveImage(
                    (camera.gt_image_normal + 1.0) / 2.0,
                    folder_path / f"gt-Normal-png/camera{i_camera}-Normal-gt.png",
                )

                if tracer_config.DATASET_IS_SYNTHETIC:
                    UTILITIES_IO.SaveExrImage(
                        camera.gt_image_depth,
                        folder_path / f"gt-Depth-exr/camera{i_camera}-Depth-gt.exr",
                    )

                    UTILITIES_IO.SaveImage(
                        (camera.gt_image_depth - camera.gt_image_depth.min())
                        / (camera.gt_image_depth.max() - camera.gt_image_depth.min()),
                        folder_path / f"gt-Depth-png/camera{i_camera}-Depth-gt.png",
                    )

                    UTILITIES_IO.SaveExrImage(
                        gt_image_albedo,
                        folder_path / f"gt-Albedo-exr/camera{i_camera}-Albedo-gt.exr",
                    )

            # [render]

            with torch.no_grad():
                render_results, time_duration = Differentiable_EAG_OptiX_nobounce(
                    camera,
                    sample_renderer,
                    self,
                )

            record_durations.append(time_duration)

            # [save Hitcount]

            UTILITIES_IO.SaveExrImage(
                render_results.image_hitcount,
                folder_path
                / f"render-Hitcount-exr/camera{i_camera}-Hitcount-render.exr",
            )

            try:

                # [save Alpha]

                psnr_alpha_linear = UTILITIES_IMAGE.Psnr(
                    image=render_results.image_alpha,
                    target=gt_image_alpha,
                )
                record_psnrs_alpha_linear.append(psnr_alpha_linear.item())

                UTILITIES_IO.SaveExrImage(
                    render_results.image_alpha,
                    folder_path
                    / f"render-Alpha-exr/camera{i_camera}-Alpha-render_psnr{psnr_alpha_linear.item():.2f}.exr",
                )

            except:

                # [250116 for Teaser]

                UTILITIES_IO.SaveExrImage(
                    render_results.image_radiance,
                    folder_path
                    / f"render-Radiance-exr/camera{i_camera}-Radiance-render_duration{time_duration:.3f}s.exr",
                )

                return

            # [save Radiance]

            # image metrics - preparation

            image_radiance = render_results.image_radiance
            image_radiance_pq = UTILITIES_COLOUR.LinearToPq(image_radiance)
            image_radiance_srgbclip = UTILITIES_COLOUR.LinearToSrgb(
                image_radiance
            ).clip(min=0.0, max=1.0)

            # image metrics - psnr

            psnr_radiance_linear = UTILITIES_IMAGE.Psnr(
                image=image_radiance,
                target=gt_image_radiance,
            )
            record_psnrs_radiance_linear.append(psnr_radiance_linear.item())

            psnr_radiance_pq = UTILITIES_IMAGE.Psnr(
                image=image_radiance_pq,
                target=gt_image_radiance_pq,
            )
            record_psnrs_radiance_pq.append(psnr_radiance_pq.item())

            psnr_radiance_srgbclip = UTILITIES_IMAGE.Psnr(
                image=image_radiance_srgbclip,
                target=gt_image_radiance_rgb_srgbclip_premultiplied,
            )
            record_psnrs_radiance_srgbclip.append(psnr_radiance_srgbclip.item())

            lpips_radiance_srgbclip = UTILITIES_IMAGE.Lpips(
                image=image_radiance_srgbclip,
                target=gt_image_radiance_rgb_srgbclip_premultiplied,
            )
            record_lpipss_radiance_srgbclip.append(lpips_radiance_srgbclip.item())

            # image metrics - flip

            # ExLog(f"{image_radiance_rgb_linear_premultiplied.shape=} {image_radiance_rgb_linear_premultiplied.min()=} {image_radiance_rgb_linear_premultiplied.max()=}")
            # ExLog(f"{image_radiance_rgb_linear_premultiplied.shape=} {gt_image_radiance_rgb_linear_premultiplied.min()=} {gt_image_radiance_rgb_linear_premultiplied.max()=}")
            flip_values, flip_value = UTILITY_FLIP.CalculateFlipValues(
                image_test=image_radiance,
                image_reference=gt_image_radiance,
            )
            record_flips_radiance_linear.append(flip_value)
            UTILITIES_IO.SaveImage(
                flip_values,
                folder_path
                / f"render-Radiance-flip-linear/camera{i_camera}-flip{flip_value:.4f}.png",
            )

            UTILITIES_IO.SaveExrImage(
                render_results.image_radiance,
                folder_path
                / f"render-Radiance-exr/camera{i_camera}-Radiance-render_duration{time_duration:.3f}s_psnrsrgbclip{psnr_radiance_srgbclip.item():.2f}_lpipssrgbclip{lpips_radiance_srgbclip.item():.4f}_flip{flip_value:.4f}.exr",
            )

            # [save Emissive]

            if gt_image_emissive != None:
                psnr_emissive_linear = UTILITIES_IMAGE.Psnr(
                    image=camera.gt_image_emissive.cuda(),
                    target=render_results.image_emissive,
                )
                record_psnrs_emissive_linear.append(psnr_emissive_linear.item())

                UTILITIES_IO.SaveExrImage(
                    render_results.image_emissive,
                    folder_path
                    / f"render-Emissive-exr/camera{i_camera}-Emissive-render_psnrlinear{psnr_emissive_linear.item():.2f}.exr",
                )
            else:
                UTILITIES_IO.SaveExrImage(
                    render_results.image_emissive,
                    folder_path
                    / f"render-Emissive-exr/camera{i_camera}-Emissive-render.exr",
                )

            render_results_image_emissive_float = (
                render_results.image_emissive > 0.10
            ).to(dtype=torch.float32)

            UTILITIES_IO.SaveExrImage(
                render_results_image_emissive_float
                * torch.tensor([0.0, 1.0, 0.0])[:, None, None]
                + (1.0 - render_results_image_emissive_float)
                * render_results.image_radiance,
                folder_path
                / f"render-Emissive-exr/camera{i_camera}-Emissive-render_w-radiances.exr",
            )

            # [save Albedo]

            if tracer_config.DATASET_IS_SYNTHETIC:
                psnr_albedo_linear = UTILITIES_IMAGE.Psnr(
                    image=camera.gt_image_albedo.cuda()
                    * (1.0 - render_results_image_emissive_float),
                    target=render_results.image_albedo
                    * (1.0 - render_results_image_emissive_float),
                )
                record_psnrs_albedo_linear.append(psnr_albedo_linear.item())

                psnr_albedo_srgbclip = UTILITIES_IMAGE.Psnr(
                    image=UTILITIES_COLOUR.LinearToSrgb(
                        (
                            camera.gt_image_albedo.cuda()
                            * (1.0 - render_results_image_emissive_float)
                        )
                    ).clip(min=0.0, max=1.0),
                    target=UTILITIES_COLOUR.LinearToSrgb(
                        (
                            render_results.image_albedo
                            * (1.0 - render_results_image_emissive_float)
                        )
                    ).clip(min=0.0, max=1.0),
                )
                record_psnrs_albedo_srgbclip.append(psnr_albedo_srgbclip.item())

                UTILITIES_IO.SaveExrImage(
                    render_results.image_albedo
                    * (1.0 - render_results_image_emissive_float),
                    folder_path
                    / f"render-Albedo-exr/camera{i_camera}-Albedo-render_psnrlinear{psnr_albedo_linear.item():.2f}_psnrsrgbclip{psnr_albedo_srgbclip.item():.2f}.exr",
                )
            else:
                UTILITIES_IO.SaveExrImage(
                    render_results.image_albedo
                    * (1.0 - render_results_image_emissive_float),
                    folder_path
                    / f"render-Albedo-exr/camera{i_camera}-Albedo-render.exr",
                )

            # [save Normal]

            UTILITIES_IO.SaveExrImage(
                render_results.image_normal,
                folder_path / f"render-Normal-exr/camera{i_camera}-Normal-render.exr",
            )
            UTILITIES_IO.SaveImage(
                (render_results.image_normal + 1.0) / 2.0,
                folder_path / f"render-Normal-png/camera{i_camera}-Normal-render.png",
            )

            # [save Distance]

            UTILITIES_IO.SaveExrImage(
                render_results.image_distance,
                folder_path
                / f"render-Distance-exr/camera{i_camera}-Distance-render.exr",
            )

            render_image_depth_normal = (
                EAGTracingResult.DeriveNormalFromDistanceGivenCamera(
                    buffer_distances=render_results.buffer_distances, camera=camera
                )
            )

            UTILITIES_IO.SaveExrImage(
                render_image_depth_normal,
                folder_path
                / f"render-DepthNormal-exr/camera{i_camera}-DepthNormal-render.exr",
            )
            UTILITIES_IO.SaveImage(
                (render_image_depth_normal + 1.0) / 2.0,
                folder_path
                / f"render-DepthNormal-png/camera{i_camera}-DepthNormal-render.png",
            )

            # [save Depth]

            render_image_depth = EAGTracingResult.DeriveDepthFromDistanceGivenCamera(
                buffer_distances=render_results.buffer_distances, camera=camera
            )

            UTILITIES_IO.SaveExrImage(
                render_image_depth,
                folder_path / f"render-Depth-exr/camera{i_camera}-Depth-render.exr",
            )
            UTILITIES_IO.SaveImage(
                (render_image_depth - render_image_depth.min())
                / (render_image_depth.max() - render_image_depth.min()),
                folder_path / f"render-Depth-png/camera{i_camera}-Depth-render.png",
            )

        record_file_path = folder_path / "_records.py"
        with open(record_file_path, "w") as f:
            f.write(f"avg_duration = {sum(record_durations)/len(record_durations)}\n")
            f.write(
                f"avg_psnr_radiance_linear = {sum(record_psnrs_radiance_linear)/len(record_psnrs_radiance_linear)}\n"
            )
            f.write(
                f"avg_psnr_radiance_pq = {sum(record_psnrs_radiance_pq)/len(record_psnrs_radiance_pq)}\n"
            )
            f.write(
                f"avg_psnr_radiance_srgbclip = {sum(record_psnrs_radiance_srgbclip)/len(record_psnrs_radiance_srgbclip)}\n"
            )
            f.write(
                f"avg_lpips_radiance_srgbclip = {sum(record_lpipss_radiance_srgbclip)/len(record_lpipss_radiance_srgbclip)}\n"
            )

            f.write(
                f"avg_flip_radiance_linear = {sum(record_flips_radiance_linear)/len(record_flips_radiance_linear)}\n"
            )
            if len(record_psnrs_emissive_linear) != 0:
                f.write(
                    f"avg_psnr_emissive_linear = {sum(record_psnrs_emissive_linear)/len(record_psnrs_emissive_linear)}\n"
                )
            if tracer_config.DATASET_IS_SYNTHETIC:
                f.write(
                    f"avg_psnr_albedo_linear = {sum(record_psnrs_albedo_linear)/len(record_psnrs_albedo_linear)}\n"
                )
                f.write(
                    f"avg_psnr_albedo_srgbclip = {sum(record_psnrs_albedo_srgbclip)/len(record_psnrs_albedo_srgbclip)}\n"
                )
            f.write(
                f"avg_psnr_alpha_linear = {sum(record_psnrs_alpha_linear)/len(record_psnrs_alpha_linear)}\n"
            )

            f.write(f"\n")

            f.write(f"durations = {record_durations}\n")
            f.write(f"psnrs_radiance_linear = {record_psnrs_radiance_linear}\n")
            f.write(f"psnrs_radiance_pq = {record_psnrs_radiance_pq}\n")
            f.write(f"psnrs_radiance_srgbclip = {record_psnrs_radiance_srgbclip}\n")
            f.write(f"lpipss_radiance_srgbclip = {record_lpipss_radiance_srgbclip}\n")
            f.write(f"psnrs_emissive_linear = {record_psnrs_emissive_linear}\n")
            if tracer_config.DATASET_IS_SYNTHETIC:
                f.write(f"psnrs_albedo_linear = {record_psnrs_albedo_linear}\n")
                f.write(f"psnrs_albedo_srgbclip = {record_psnrs_albedo_srgbclip}\n")
            f.write(f"flips_radiance_linear = {record_flips_radiance_linear}\n")
            f.write(f"psnrs_alpha_linear = {record_psnrs_alpha_linear}\n")

        ExLog(f"Saved records at {record_file_path}.")

    @torch.no_grad()
    def saveSingleBounceResultsOnCameras(
        self,
        cameras: list[EAGCamera],
        tracer_config: TracerConfig,
        folder_path: pathlib.Path = None,
        spp: int = 8,
    ):
        if folder_path == None:
            folder_path = tracer_config.OUTPUT_FOLDER_PATH / f"single-bounce-spp{spp}"

        # [optix backend]

        sample_renderer = eag_pt_tracer_optix.SampleRenderer()
        triangles_vertices, triangles_indices = (
            self.get_triangles_vertices_and_indices_for_optix_build_acceleration_structure()
        )
        sample_renderer.buildAccel(
            triangles_vertices.shape[0],
            triangles_vertices.contiguous().data_ptr(),
            triangles_indices.shape[0],
            triangles_indices.contiguous().data_ptr(),
        )

        # [records]

        record_durations: list[float] = []
        record_denoised_gamma_psnrs: list[float] = []
        record_denoised_gamma_lpipss: list[float] = []
        record_denoised_linear_flips: list[float] = []

        for i_camera, camera in enumerate(cameras):

            with torch.no_grad():
                pixels_rendering_radiances, duration = (
                    Differentiable_EAG_OptiX_singlebounce(
                        camera, spp, sample_renderer, self
                    )
                )

            record_durations.append(duration)

            image_averaged_noisy = einops.rearrange(
                pixels_rendering_radiances, "p c -> c p"
            ).reshape((3, camera.image_height, camera.image_width))

            # [save image]

            # gt

            gt_image_linear = camera.gt_image_radiance_rgb_linear_premultiplied.cuda()
            UTILITIES_IO.SaveExrImage(
                gt_image_linear,
                path=folder_path / f"camera{i_camera}_gt-radiance_rgb.exr",
            )

            # noisy

            psnr_gamma_rendered_noisy = UTILITIES_IMAGE.Psnr(
                UTILITIES_COLOUR.LinearToSrgb(image_averaged_noisy).clip(
                    min=0.0, max=1.0
                ),
                UTILITIES_COLOUR.LinearToSrgb(gt_image_linear).clip(min=0.0, max=1.0),
            )
            UTILITIES_IO.SaveExrImage(
                image_averaged_noisy,
                path=folder_path
                / f"camera{i_camera}_spp{spp}_radiance_noisy_psnrsrgbclip{psnr_gamma_rendered_noisy.item():.2f}.exr",
            )

            # denoised

            image_averaged_denoised = UTILITIES_IMAGE.Denoise(
                image=image_averaged_noisy
            )

            image_render_denoised_gamma_clipped = UTILITIES_COLOUR.LinearToSrgb(
                image_averaged_denoised
            ).clip(min=0.0, max=1.0)
            image_gt_gamma_clipped = UTILITIES_COLOUR.LinearToSrgb(
                gt_image_linear
            ).clip(min=0.0, max=1.0)

            # [psnr]
            psnr_gamma_rendered_denoised = UTILITIES_IMAGE.Psnr(
                image_render_denoised_gamma_clipped,
                image_gt_gamma_clipped,
            )
            record_denoised_gamma_psnrs.append(psnr_gamma_rendered_denoised.item())
            # [lpips]
            lpips_gamma_rendered_denoised = UTILITIES_IMAGE.Lpips(
                image_render_denoised_gamma_clipped,
                image_gt_gamma_clipped,
            )
            record_denoised_gamma_lpipss.append(lpips_gamma_rendered_denoised.item())
            # [flip]
            image_flip_values, flip_linear_rendered_denoised = (
                UTILITY_FLIP.CalculateFlipValues(
                    image_test=image_averaged_denoised,
                    image_reference=gt_image_linear,
                )
            )
            record_denoised_linear_flips.append(flip_linear_rendered_denoised)
            UTILITIES_IO.SaveImage(
                image_flip_values,
                folder_path
                / f"camera{i_camera}_spp{spp}_fliplinear{flip_linear_rendered_denoised:.4f}.png",
            )

            UTILITIES_IO.SaveExrImage(
                image_averaged_denoised,
                path=folder_path
                / f"camera{i_camera}_spp{spp}_radiance_duration{duration:.3f}_denoised_psnrsrgbclip{psnr_gamma_rendered_denoised.item():.2f}_lpipssrgbclip{lpips_gamma_rendered_denoised.item():.4f}_fliplinear{flip_linear_rendered_denoised:.4f}.exr",
            )

        record_file_path = folder_path / "_records.py"

        with open(record_file_path, "w") as f:
            f.write(f"avg_duration = {sum(record_durations)/len(record_durations)}\n")
            f.write(
                f"avg_denoised_gamma_psnrs = {sum(record_denoised_gamma_psnrs)/len(record_denoised_gamma_psnrs)}\n"
            )

            f.write(
                f"avg_denoised_gamma_lpipss = {sum(record_denoised_gamma_lpipss)/len(record_denoised_gamma_lpipss)}\n"
            )
            f.write(
                f"avg_denoised_linear_flips = {sum(record_denoised_linear_flips)/len(record_denoised_linear_flips)}\n"
            )

            f.write(f"\n")

            f.write(f"durations = {record_durations}\n")
            f.write(f"denoised_gamma_psnrs = {record_denoised_gamma_psnrs}\n")

            f.write(f"denoised_gamma_lpipss = {record_denoised_gamma_lpipss}\n")
            f.write(f"denoised_linear_flips = {record_denoised_linear_flips}\n")

    @torch.no_grad()
    def savePathTracingResultsOnCameras(
        self,
        tracer_config: TracerConfig,
        cameras: list[EAGCamera],
        spp: int = 8,
        bounce_limit: int = 8,
        folder_path: pathlib.Path = None,
    ):
        if folder_path == None:
            folder_path = tracer_config.OUTPUT_FOLDER_PATH / "path-tracing"

        # [optix backend]

        sample_renderer = eag_pt_tracer_optix.SampleRenderer()
        triangles_vertices, triangles_indices = (
            self.get_triangles_vertices_and_indices_for_optix_build_acceleration_structure()
        )
        sample_renderer.buildAccel(
            triangles_vertices.shape[0],
            triangles_vertices.contiguous().data_ptr(),
            triangles_indices.shape[0],
            triangles_indices.contiguous().data_ptr(),
        )

        # [records]

        record_durations: list[float] = []
        record_denoised_gamma_psnrs: list[float] = []
        record_denoised_gamma_lpipss: list[float] = []
        record_denoised_linear_flips: list[float] = []

        for i_camera, camera in enumerate(cameras):
            # if i_camera in [26, 23, 33, 15, 27]: # [my-box]
            # middle (26)
            # left (23), right (33)
            # left back (15), right back (27)

            # if i_camera in [8, 9, 14, 20, 29]:  # [FR-classroom]

            rays = camera.generateRays()

            pixels_rendering_radiances = torch.zeros(
                (camera.image_height * camera.image_width, 3),
                dtype=torch.float32,
                device="cuda",
            )

            with ExTimer(f"path tracing, spp {spp}", enable=True):
                timer = ExTimer("tmp")
                sample_renderer.pathtracing(
                    # [numbers]
                    camera.image_height,
                    camera.image_width,
                    bounce_limit,  # bounce_limit
                    spp,  # spp
                    # [input - surfels]
                    self.positions.contiguous().data_ptr(),
                    self.scales.contiguous().data_ptr(),
                    self.quaternions.contiguous().data_ptr(),
                    self.opacities.contiguous().data_ptr(),
                    self.radiances.contiguous().data_ptr(),
                    self.emissives.contiguous().data_ptr(),
                    self.albedos.contiguous().data_ptr(),
                    # [input - rays]
                    rays.origins.contiguous().data_ptr(),
                    rays.directions.contiguous().data_ptr(),
                    # [output - results]
                    pixels_rendering_radiances.contiguous().data_ptr(),
                )
                duration = timer.stop()
            record_durations.append(duration)

            image_averaged_noisy = einops.rearrange(
                pixels_rendering_radiances, "p c -> c p"
            ).reshape((3, camera.image_height, camera.image_width))

            # [save image]

            # gt

            gt_image_linear = camera.gt_image_radiance_rgb_linear_premultiplied.cuda()
            UTILITIES_IO.SaveExrImage(
                gt_image_linear,
                path=folder_path / f"camera{i_camera}_gt-radiance_rgb.exr",
            )

            try:

                # noisy

                psnr_gamma_rendered_noisy = UTILITIES_IMAGE.Psnr(
                    UTILITIES_COLOUR.LinearToSrgb(image_averaged_noisy).clip(
                        min=0.0, max=1.0
                    ),
                    UTILITIES_COLOUR.LinearToSrgb(gt_image_linear).clip(
                        min=0.0, max=1.0
                    ),
                )
                UTILITIES_IO.SaveExrImage(
                    image_averaged_noisy,
                    path=folder_path
                    / f"camera{i_camera}_spp{spp}_radiance_noisy_psnrsrgbclip{psnr_gamma_rendered_noisy.item():.2f}.exr",
                )

            except:

                # [260116 for Teaser]

                UTILITIES_IO.SaveExrImage(
                    image_averaged_noisy,
                    path=folder_path / f"camera{i_camera}_spp{spp}_radiance_noisy.exr",
                )

                image_averaged_denoised = UTILITIES_IMAGE.Denoise(
                    image=image_averaged_noisy
                )

                UTILITIES_IO.SaveExrImage(
                    image_averaged_denoised,
                    path=folder_path
                    / f"camera{i_camera}_spp{spp}_radiance_duration{duration:.3f}_denoised.exr",
                )

                ExLog(f"Saved {spp=}", "Teaser")

                return

            # denoised

            image_averaged_denoised = UTILITIES_IMAGE.Denoise(
                image=image_averaged_noisy
            )
            psnr_linear_rendered_denoised = UTILITIES_IMAGE.Psnr(
                image_averaged_denoised, gt_image_linear
            )
            image_render_denoised_gamma_clipped = UTILITIES_COLOUR.LinearToSrgb(
                image_averaged_denoised
            ).clip(min=0.0, max=1.0)
            image_gt_gamma_clipped = UTILITIES_COLOUR.LinearToSrgb(
                gt_image_linear
            ).clip(min=0.0, max=1.0)

            psnr_gamma_rendered_denoised = UTILITIES_IMAGE.Psnr(
                image_render_denoised_gamma_clipped,
                image_gt_gamma_clipped,
            )
            record_denoised_gamma_psnrs.append(psnr_gamma_rendered_denoised.item())

            lpips_gamma_rendered_denoised = UTILITIES_IMAGE.Lpips(
                image_render_denoised_gamma_clipped,
                image_gt_gamma_clipped,
            )
            record_denoised_gamma_lpipss.append(lpips_gamma_rendered_denoised.item())

            image_flip_values, flip_linear_rendered_denoised = (
                UTILITY_FLIP.CalculateFlipValues(
                    image_test=image_averaged_denoised,
                    image_reference=gt_image_linear,
                )
            )
            record_denoised_linear_flips.append(flip_linear_rendered_denoised)
            UTILITIES_IO.SaveImage(
                image_flip_values,
                folder_path
                / f"camera{i_camera}_spp{spp}_fliplinear{flip_linear_rendered_denoised:.4f}.png",
            )

            UTILITIES_IO.SaveExrImage(
                image_averaged_denoised,
                path=folder_path
                / f"camera{i_camera}_spp{spp}_radiance_duration{duration:.3f}_denoised_psnrsrgbclip{psnr_gamma_rendered_denoised.item():.2f}_lpipssrgbclip{lpips_gamma_rendered_denoised.item():.4f}_fliplinear{flip_linear_rendered_denoised:.4f}.exr",
            )

        record_file_path = folder_path / "_records.py"

        with open(record_file_path, "w") as f:
            f.write(f"avg_duration = {sum(record_durations)/len(record_durations)}\n")
            f.write(
                f"avg_denoised_gamma_psnrs = {sum(record_denoised_gamma_psnrs)/len(record_denoised_gamma_psnrs)}\n"
            )

            f.write(
                f"avg_denoised_gamma_lpipss = {sum(record_denoised_gamma_lpipss)/len(record_denoised_gamma_lpipss)}\n"
            )
            f.write(
                f"avg_denoised_linear_flips = {sum(record_denoised_linear_flips)/len(record_denoised_linear_flips)}\n"
            )

            f.write(f"\n")

            f.write(f"durations = {record_durations}\n")
            f.write(f"denoised_gamma_psnrs = {record_denoised_gamma_psnrs}\n")

            f.write(f"denoised_gamma_lpipss = {record_denoised_gamma_lpipss}\n")
            f.write(f"denoised_linear_flips = {record_denoised_linear_flips}\n")


class LearnableEmissionAwareGaussians(torch.nn.Module):

    def ActivationScales(x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(x)

    def InverseActivationScales(y: torch.Tensor) -> torch.Tensor:
        return torch.log(y / (1 - y))

    def ActivationQauternions(x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.normalize(x)

    def ActivationOpacities(x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(x)

    def InverseActivationOpacities(y: torch.Tensor) -> torch.Tensor:
        return torch.log(y / (1 - y))

    def ActivationRadiances(x: torch.Tensor) -> torch.Tensor:
        return 100.0 * torch.sigmoid(x)

    def InverseActivationRadiance(y: torch.Tensor) -> torch.Tensor:
        return torch.log((y / 100.0) / (1 - (y / 100.0)))

    def ActivationEmissives(x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(x)

    def InverseActivationEmissives(y: torch.Tensor) -> torch.Tensor:
        return torch.log(y / (1 - y))

    def ActivationAlbedos(x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(x)

    def InverseActivationAlbedos(y: torch.Tensor) -> torch.Tensor:
        return torch.log(y / (1 - y))

    def __init__(
        self,
        gaussians: EmissionAwareGaussians,
        nvs_dataset: EAGNvsDataset,
    ) -> None:
        """
        pass in initialized Gaussians. LearnableGaussians only counts for optimization with a fixed number of Gaussians.
        """
        super().__init__()

        self.count: int = gaussians.count

        self.nvs_dataset: EAGNvsDataset = nvs_dataset

        if gaussians.positions.isnan().any():
            ExLog("LearnableGaussian.__init__() has nan!!!", "ERROR")
            exit(-1)

        # ExLog("clip scale to [0.01, 0.99] to avoid exploded gradient dL_dscale")
        gaussians.scales = gaussians.scales.clip(min=0.000001, max=0.999999)

        with torch.no_grad():
            self.parameters_positions: torch.Tensor = torch.nn.Parameter(
                gaussians.positions.clone(),
                requires_grad=True,
            )
            self.parameters_scales: torch.Tensor = torch.nn.Parameter(
                LearnableEmissionAwareGaussians.InverseActivationScales(
                    gaussians.scales.clone()
                ),
                requires_grad=True,
            )
            self.parameters_quaternions: torch.Tensor = torch.nn.Parameter(
                gaussians.quaternions.clone(),
                requires_grad=True,
            )
            self.parameters_opacities: torch.Tensor = torch.nn.Parameter(
                LearnableEmissionAwareGaussians.InverseActivationOpacities(
                    gaussians.opacities.clone()
                ),
                requires_grad=True,
            )
            self.parameters_radiances: torch.Tensor = torch.nn.Parameter(
                LearnableEmissionAwareGaussians.InverseActivationRadiance(
                    gaussians.radiances.clone()
                ),
                requires_grad=True,
            )

            self.parameters_emissives: torch.Tensor = torch.nn.Parameter(
                LearnableEmissionAwareGaussians.InverseActivationEmissives(
                    gaussians.emissives.clone()
                ),
                requires_grad=True,
            )

            self.parameters_albedos: torch.Tensor = torch.nn.Parameter(
                LearnableEmissionAwareGaussians.InverseActivationAlbedos(
                    gaussians.albedos.clone()
                ),
                requires_grad=True,
            )

    @property
    def positions(self) -> torch.Tensor:
        return self.parameters_positions

    @property
    def scales(self) -> torch.Tensor:
        return LearnableEmissionAwareGaussians.ActivationScales(self.parameters_scales)

    @property
    def quaternions(self) -> torch.Tensor:
        return LearnableEmissionAwareGaussians.ActivationQauternions(
            self.parameters_quaternions
        )

    @property
    def opacities(self) -> torch.Tensor:
        return LearnableEmissionAwareGaussians.ActivationOpacities(
            self.parameters_opacities
        )

    @property
    def radiances(self) -> torch.Tensor:
        return LearnableEmissionAwareGaussians.ActivationRadiances(
            self.parameters_radiances
        )

    @property
    def emissives(self) -> torch.Tensor:
        return LearnableEmissionAwareGaussians.ActivationEmissives(
            self.parameters_emissives
        )

    @property
    def albedos(self) -> torch.Tensor:
        return LearnableEmissionAwareGaussians.ActivationAlbedos(
            self.parameters_albedos
        )

    def get_triangles_vertices_and_indices_for_optix_build_acceleration_structure(
        self, *args, **kwargs
    ):
        return EmissionAwareGaussians.get_triangles_vertices_and_indices_for_optix_build_acceleration_structure(
            self, *args, **kwargs
        )

    def train(self, tracer_config: TracerConfig, cameras: list[EAGCamera]) -> None:

        # cameras = self.nvs_dataset.train_set_cameras

        # https://pytorch.org/docs/stable/optim.html
        parameters = [
            {
                "params": [self.parameters_positions],
                "lr": tracer_config.LEARNING_RATE_POSITION
                * tracer_config.LEARNING_RATE_MULTIPLE,
            },
            {
                "params": [self.parameters_scales],
                "lr": tracer_config.LEARNING_RATE_SCALE
                * tracer_config.LEARNING_RATE_MULTIPLE,
            },
            {
                "params": [self.parameters_quaternions],
                "lr": tracer_config.LEARNING_RATE_QUATERNION
                * tracer_config.LEARNING_RATE_MULTIPLE,
            },
            {
                "params": [self.parameters_opacities],
                "lr": tracer_config.LEARNING_RATE_OPACITY
                * tracer_config.LEARNING_RATE_MULTIPLE,
            },
            {
                "params": [self.parameters_radiances],
                "lr": tracer_config.LEARNING_RATE_RADIANCE
                * tracer_config.LEARNING_RATE_MULTIPLE,
            },
            {
                "params": [self.parameters_emissives],
                "lr": tracer_config.LEARNING_RATE_EMISSIVE
                * tracer_config.LEARNING_RATE_MULTIPLE,
            },
        ]
        optimizer = torch.optim.Adam(parameters, lr=0.0)

        record_forward_durations = []
        record_backward_durations = []

        # [prepare SampleRenderer()]

        sample_renderer = eag_pt_tracer_optix.SampleRenderer()

        # for iter in tqdm.tqdm(range(tracer_config.ITERATION + 1)):
        for iter in range(tracer_config.ITERATION + 1):

            if iter % 100 == 0:
                ExLog(f"Training... iter{iter}")

            if iter == 0 and tracer_config.SAVE_ITER0_RENDERS:

                with torch.no_grad():

                    initial_gaussians: EmissionAwareGaussians = self.toGaussians()

                    ExLog(f"Saving iter0 results...")

                    initial_gaussians.savePly(
                        path=tracer_config.OUTPUT_FOLDER_PATH
                        / f"iter0-plys/optimized-2d-gaussians.ply"
                    )
                    initial_gaussians.saveColoredPointCloud(
                        path=tracer_config.OUTPUT_FOLDER_PATH
                        / f"iter0-plys/colored-point-cloud.ply"
                    )
                    initial_gaussians.saveColoredPointCloudEmissivesWithThreshold(
                        path=tracer_config.OUTPUT_FOLDER_PATH
                        / f"iter0-plys/colored-point-cloud_highlight-emissive-parts.ply"
                    )
                    initial_gaussians.saveColoredBoundingRectangles(
                        path=tracer_config.OUTPUT_FOLDER_PATH
                        / f"iter0-plys/colored-bounding-rectangles_scale-multiple-1.0.ply"
                    )
                    initial_gaussians.saveTsdfMeshFromDepthAndRgb(
                        tracer_config=tracer_config,
                        train_set_cameras=self.nvs_dataset.train_set_cameras,
                        folder=tracer_config.OUTPUT_FOLDER_PATH / f"iter0-plys",
                    )

                    initial_gaussians.saveNoBounceResultsOnCameras(
                        tracer_config=tracer_config,
                        cameras=self.nvs_dataset.test_set_cameras,
                        folder_path=tracer_config.OUTPUT_FOLDER_PATH / f"iter0-images",
                        is_to_save_groundtruths=True,
                    )

                    ExLog(f"Saved iter0 results.")

            # [rebuild scene, since surfels were updated]

            with torch.no_grad():

                triangles_vertices, triangles_indices = (
                    self.get_triangles_vertices_and_indices_for_optix_build_acceleration_structure()
                )

                sample_renderer.buildAccel(
                    triangles_vertices.shape[0],
                    triangles_vertices.contiguous().data_ptr(),
                    triangles_indices.shape[0],
                    triangles_indices.contiguous().data_ptr(),
                )

            # [pick camera]

            random_i_train_camera = random.randint(0, len(cameras) - 1)
            random_train_camera: EAGCamera = cameras[random_i_train_camera]

            # [prepare gt for further usage]

            # not pre-multiplied
            if random_train_camera.path_tracing_radiance_rgb_linear == None:
                # [common training process using gt]
                image_rgbs_linear_premultiplied_gt = (
                    random_train_camera.gt_image_radiance_rgb_linear_premultiplied.cuda()
                )
            else:
                # [light baking process using re-loaded path tracing rendering]
                image_rgbs_linear_premultiplied_gt = (
                    random_train_camera.path_tracing_radiance_rgb_linear.cuda()
                )

            image_rgbs_pq_premultiplied_gt = UTILITIES_COLOUR.LinearToPq(
                image_rgbs_linear_premultiplied_gt
            )
            image_as_gt = random_train_camera.gt_image_alpha.cuda()
            if random_train_camera.gt_image_emissive != None:
                image_emissives_0or1_gt = random_train_camera.gt_image_emissive.cuda()

            # [call optix backend]

            (
                render_results,
                record_forward_duration,
            ) = Differentiable_EAG_OptiX_nobounce(
                random_train_camera, sample_renderer, self
            )
            record_forward_durations.append(record_forward_duration)

            # [loss - rgb]

            # add black background
            loss_l1: torch.Tensor = UTILITIES_IMAGE.L1Loss(
                image=UTILITIES_COLOUR.LinearToPq(render_results.image_radiance),
                target=image_rgbs_pq_premultiplied_gt,
            )
            ssim: torch.Tensor = UTILITIES_IMAGE.Ssim(
                image=UTILITIES_COLOUR.LinearToPq(render_results.image_radiance),
                target=image_rgbs_pq_premultiplied_gt,
            )
            loss_dssim: torch.Tensor = 1.0 - ssim

            # [loss - emissive]

            if random_train_camera.gt_image_emissive != None:
                loss_emissives = torch.abs(
                    (render_results.image_emissive - image_emissives_0or1_gt)
                ).mean()
            else:
                loss_emissives = torch.tensor([0.0])

            # [loss - alpha]

            loss_alpha_channel_supervision = (
                UTILITIES_IMAGE.L1Loss(render_results.image_alpha, image_as_gt)
                # if iter > 3000
                # else torch.tensor([0.0])
            )

            # [loss - normal supervision]

            loss_normal_supervision = (
                (
                    1.0
                    - (
                        render_results.image_normal
                        * random_train_camera.gt_image_normal.cuda()
                    ).sum(dim=0, keepdim=True)
                ).mean()
                if iter > tracer_config.NORMAL_SUPERVISION_START_FROM_ITER
                else torch.tensor([0.0])
            )

            # [loss - normal consistency]

            image_depth_normal = EAGTracingResult.DeriveNormalFromDistanceGivenCamera(
                buffer_distances=render_results.buffer_distances,
                camera=random_train_camera,
            )

            loss_consistency_between_direct_normal_and_depth_normal = (
                (
                    1.0
                    - (
                        (
                            image_depth_normal
                            * random_train_camera.gt_image_normal.cuda()
                        )[:, 1:-1, 1:-1]
                    ).sum(dim=0, keepdim=True)
                ).mean()
                if iter > tracer_config.NORMAL_CONSISTENCY_START_FROM_ITER
                else torch.tensor([0.0])
            )

            # [loss - depth supervision for synthetic scene]

            if tracer_config.DATASET_IS_SYNTHETIC:

                render_image_depth = (
                    EAGTracingResult.DeriveDepthFromDistanceGivenCamera(
                        buffer_distances=render_results.buffer_distances,
                        camera=random_train_camera,
                    )
                )

                loss_depth_supervision = (
                    UTILITIES_IMAGE.L1Loss(
                        render_image_depth,
                        random_train_camera.gt_image_depth.cuda(),
                    )
                    if iter > tracer_config.DEPTH_SUPERVISION_START_FROM_ITER
                    else torch.tensor([0.0])
                )

            # [full loss: \mathcal{L}]

            loss: torch.Tensor = (
                (
                    loss_l1 * (1.0 - tracer_config.LOSS_LAMBDA_DSSIM)
                    + loss_dssim * tracer_config.LOSS_LAMBDA_DSSIM
                )
                * tracer_config.LOSS_OVERALL_RADIANCE
                + loss_emissives * tracer_config.LOSS_EMISSIVE_MASK_SUPERVISION
                + loss_alpha_channel_supervision
                * tracer_config.LOSS_ALPHA_CHANNEL_SUPERVISION
                + loss_normal_supervision * tracer_config.LOSS_NORMAL_SUPERVISION
                + (
                    loss_depth_supervision * tracer_config.LOSS_DEPTH_SUPERVISION
                    if tracer_config.DATASET_IS_SYNTHETIC
                    else 0.0
                )
                + loss_consistency_between_direct_normal_and_depth_normal
                * tracer_config.LOSS_NORMAL_CONSISTENCY
            )

            # [backward]

            time_backward_start = time.perf_counter()

            optimizer.zero_grad()
            loss.backward(retain_graph=True)

            # print()
            # ExLog(f"[DEBUG] {iter=} {self.parameters_scales.grad.min()=} {self.parameters_scales.grad.max()=} {self.parameters_scales.grad.mean()=} {self.parameters_scales.grad.median()=}")

            if torch.isnan(self.parameters_positions._grad).sum() != 0:
                ExLog(f"{iter=} {torch.isnan(self.parameters_positions._grad).sum()=}")
                breakpoint()
                exit()
            if torch.isnan(self.parameters_scales._grad).sum() != 0:
                ExLog(f"{iter=} {torch.isnan(self.parameters_scales._grad).sum()=}")
                breakpoint()
                exit()
            if torch.isnan(self.parameters_quaternions._grad).sum() != 0:
                ExLog(
                    f"{iter=} {torch.isnan(self.parameters_quaternions._grad).sum()=}"
                )
                breakpoint()
                exit()
            if torch.isnan(self.parameters_opacities._grad).sum() != 0:
                ExLog(f"{iter=} {torch.isnan(self.parameters_opacities._grad).sum()=}")
                breakpoint()
                exit()
            if torch.isnan(self.parameters_radiances._grad).sum() != 0:
                ExLog(f"{iter=} {torch.isnan(self.parameters_radiances._grad).sum()=}")
                breakpoint()
                exit()
            if torch.isnan(self.parameters_emissives._grad).sum() != 0:
                ExLog(f"{iter=} {torch.isnan(self.parameters_emissives._grad).sum()=}")
                breakpoint()
                exit()

            optimizer.step()

            time_backward_end = time.perf_counter()
            record_backward_durations.append(time_backward_end - time_backward_start)

            # [save results]

            with torch.no_grad():
                # if iter % 10 == 0:
                #     optimized_gaussians = self.toGaussians()

                #     ExLog(
                #         f"{optimized_gaussians.positions.min().item()=} {optimized_gaussians.positions.max().item()=} {optimized_gaussians.positions.mean().item()=} {optimized_gaussians.positions.median().item()=}"
                #     )
                #     ExLog(
                #         f"{optimized_gaussians.scales.min().item()=} {optimized_gaussians.scales.max().item()=} {optimized_gaussians.scales.mean().item()=} {optimized_gaussians.scales.median().item()=}"
                #     )
                #     ExLog(
                #         f"{optimized_gaussians.quaternions.min().item()=} {optimized_gaussians.quaternions.max().item()=} {optimized_gaussians.quaternions.mean().item()=} {optimized_gaussians.quaternions.median().item()=}"
                #     )

                #     ExLog(
                #         f"{optimized_gaussians.opacities.min().item()=} {optimized_gaussians.opacities.max().item()=} {optimized_gaussians.opacities.mean().item()=} {optimized_gaussians.opacities.median().item()=}"
                #     )
                #     ExLog(
                #         f"{optimized_gaussians.radiances.min().item()=} {optimized_gaussians.radiances.max().item()=} {optimized_gaussians.radiances.mean().item()=} {optimized_gaussians.radiances.median().item()=}"
                #     )

                if iter in tracer_config.ITERATIONS_TO_SAVE:
                    optimized_gaussians = self.toGaussians()

                    optimized_gaussians.savePly(
                        path=tracer_config.OUTPUT_FOLDER_PATH
                        / f"iter{iter}-plys/optimized-2d-gaussians.ply"
                    )
                    optimized_gaussians.saveColoredPointCloud(
                        path=tracer_config.OUTPUT_FOLDER_PATH
                        / f"iter{iter}-plys/colored-point-cloud.ply"
                    )
                    optimized_gaussians.saveColoredPointCloudEmissivesWithThreshold(
                        path=tracer_config.OUTPUT_FOLDER_PATH
                        / f"iter{iter}-plys/colored-point-cloud_highlight-emissive-parts.ply"
                    )
                    optimized_gaussians.saveColoredBoundingRectangles(
                        path=tracer_config.OUTPUT_FOLDER_PATH
                        / f"iter{iter}-plys/colored-bounding-rectangles_scale-multiple-1.0.ply"
                    )
                    optimized_gaussians.saveTsdfMeshFromDepthAndRgb(
                        tracer_config=tracer_config,
                        train_set_cameras=self.nvs_dataset.train_set_cameras,
                        folder=tracer_config.OUTPUT_FOLDER_PATH / f"iter{iter}-plys",
                    )

                if iter in tracer_config.ITERATIONS_TO_EVALUATE:
                    optimized_gaussians: EmissionAwareGaussians = self.toGaussians()

                    optimized_gaussians.saveNoBounceResultsOnCameras(
                        tracer_config=tracer_config,
                        cameras=self.nvs_dataset.test_set_cameras,
                        folder_path=tracer_config.OUTPUT_FOLDER_PATH
                        / f"iter{iter}-images",
                    )

        record_file_path = tracer_config.OUTPUT_FOLDER_PATH / "optimization/_records.py"
        record_file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(record_file_path, "w") as f:
            f.write(
                f"avg_forward_durations = {sum(record_forward_durations)/len(record_forward_durations)}\n"
            )
            f.write(
                f"avg_backward_durations = {sum(record_backward_durations)/len(record_backward_durations)}\n"
            )
            f.write(f"\n")
            f.write(f"forward_durations = {record_forward_durations}\n")
            f.write(f"backward_durations = {record_backward_durations}\n")
        ExLog(f"Saved records at {record_file_path}.")

    def optimizeAlbedosUsingSingleBounceIntoRadianceCache(
        self, tracer_config: TracerConfig
    ) -> None:
        # https://pytorch.org/docs/stable/optim.html
        parameters = [
            {
                "params": [self.parameters_albedos],
                "lr": tracer_config.LEARNING_RATE_ALBEDO
                * tracer_config.LEARNING_RATE_MULTIPLE,
            },
        ]
        optimizer = torch.optim.Adam(parameters, lr=0.0)

        record_forward_durations = []
        record_backward_durations = []

        # [prepare optix backend and build acceleration structure. no need to rebuild the scene when optimizing albedos]

        sample_renderer = eag_pt_tracer_optix.SampleRenderer()
        with torch.no_grad():
            # [rebuild scene, since surfels were updated]
            triangles_vertices, triangles_indices = (
                self.get_triangles_vertices_and_indices_for_optix_build_acceleration_structure()
            )
            sample_renderer.buildAccel(
                triangles_vertices.shape[0],
                triangles_vertices.contiguous().data_ptr(),
                triangles_indices.shape[0],
                triangles_indices.contiguous().data_ptr(),
            )

        # for iter in tqdm.tqdm(range(tracer_config.ITERATION + 1)):
        for iter in range(tracer_config.ITERATION_OPTIMIZE_ALBEDO + 1):
            if iter % 10 == 0:
                ExLog(f"Training... iter{iter}")

            with torch.no_grad():
                if iter == 0 and tracer_config.SAVE_ITER0_RENDERS:
                    ExLog(f"Saving first iter loaded results...")
                    optimized_gaussians: EmissionAwareGaussians = self.toGaussians()
                    optimized_gaussians.saveNoBounceResultsOnCameras(
                        tracer_config=tracer_config,
                        cameras=self.nvs_dataset.test_set_cameras,
                        folder_path=tracer_config.OUTPUT_FOLDER_PATH / f"iter0-images",
                        is_to_save_groundtruths=True,
                    )
                    ExLog(f"Saved first iter loaded results.")

            # random_train_camera: EAGCamera = self.nvs_dataset.test_set_cameras[0]

            # [pick camera]

            random_i_train_camera = random.randint(
                0, len(self.nvs_dataset.train_set_cameras) - 1
            )
            random_train_camera: EAGCamera = self.nvs_dataset.train_set_cameras[
                random_i_train_camera
            ]

            # [derive images]

            # not pre-multiplied
            image_rgbs_linear_premultiplied_gt = (
                random_train_camera.gt_image_radiance_rgb_linear_premultiplied.cuda()
            )

            pixels_rendering_radiances, duration = (
                Differentiable_EAG_OptiX_singlebounce(
                    camera=random_train_camera,
                    spp=tracer_config.N_SPP_OPTIMIZE_ALBEDO,
                    sample_renderer=sample_renderer,
                    surfels=self,
                )
            )

            record_forward_durations.append(duration)

            image_rgbs_linear_single_bounce_from_radiance_cache = einops.rearrange(
                pixels_rendering_radiances, "p c -> c p"
            ).reshape(
                (
                    3,
                    random_train_camera.image_height,
                    random_train_camera.image_width,
                )
            )

            image_radiance_pq_render = UTILITIES_COLOUR.LinearToPq(
                image_rgbs_linear_single_bounce_from_radiance_cache
            )
            image_radiance_pq_gt = UTILITIES_COLOUR.LinearToPq(
                image_rgbs_linear_premultiplied_gt
            )

            # [calculate loss]

            # add black background
            loss_l1: torch.Tensor = UTILITIES_IMAGE.L1Loss(
                image=image_radiance_pq_render,
                target=image_radiance_pq_gt,
            )
            ssim: torch.Tensor = UTILITIES_IMAGE.Ssim(
                image=image_radiance_pq_render,
                target=image_radiance_pq_gt,
            )
            loss_dssim: torch.Tensor = 1.0 - ssim

            loss: torch.Tensor = (
                1.0 - tracer_config.LOSS_LAMBDA_DSSIM
            ) * loss_l1 + tracer_config.LOSS_LAMBDA_DSSIM * loss_dssim

            # [backward]

            time_backward_start = time.perf_counter()

            optimizer.zero_grad()
            loss.backward()

            if pixels_rendering_radiances.isnan().sum() != 0:
                ExLog(f"{iter=} {pixels_rendering_radiances.isnan().sum()=}")
                exit()
            if torch.isnan(self.parameters_albedos._grad).sum() != 0:
                ExLog(f"{iter=} {torch.isnan(self.parameters_albedos._grad).sum()=}")
                exit()
            if torch.isnan(self.parameters_albedos).sum() != 0:
                ExLog(f"{iter=} {torch.isnan(self.parameters_albedos).sum()=}")
                exit()

            optimizer.step()

            time_backward_end = time.perf_counter()
            record_backward_durations.append(time_backward_end - time_backward_start)

            # [save results]

            with torch.no_grad():
                if iter in tracer_config.ITERATIONS_TO_SAVE_OPTIMIZE_ALBEDO:
                    optimized_gaussians = self.toGaussians()

                    optimized_gaussians.savePly(
                        path=tracer_config.OUTPUT_FOLDER_PATH
                        / f"iter{iter}-plys/optimized-2d-gaussians_iter{iter}.ply"
                    )

                if iter in tracer_config.ITERATIONS_TO_EVALUATE_OPTIMIZE_ALBEDO:
                    optimized_gaussians = self.toGaussians()

                    # optimized_gaussians.saveSingleBounceResultsOnCameras(
                    #     cameras=self.nvs_dataset.test_set_cameras,
                    #     folder_path=tracer_config.OUTPUT_FOLDER_PATH
                    #     / f"images_single-bounce_optimized_albedos_iter{iter}",
                    #     tracer_config=tracer_config,
                    # )

                    optimized_gaussians.saveNoBounceResultsOnCameras(
                        tracer_config=tracer_config,
                        cameras=self.nvs_dataset.test_set_cameras,
                        folder_path=tracer_config.OUTPUT_FOLDER_PATH
                        / f"iter{iter}-images",
                    )

        record_file_path = tracer_config.OUTPUT_FOLDER_PATH / "optimization/_records.py"
        record_file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(record_file_path, "w") as f:
            f.write(
                f"avg_forward_durations = {sum(record_forward_durations)/len(record_forward_durations)}\n"
            )
            f.write(
                f"avg_backward_durations = {sum(record_backward_durations)/len(record_backward_durations)}\n"
            )
            f.write(f"\n")
            f.write(f"forward_durations = {record_forward_durations}\n")
            f.write(f"backward_durations = {record_backward_durations}\n")
        ExLog(f"Saved records at {record_file_path}.")

    def toGaussians(self) -> EmissionAwareGaussians:
        positions = self.positions.clone().detach().requires_grad_(False)
        scales = self.scales.clone().detach().requires_grad_(False)
        quaternions = self.quaternions.clone().detach().requires_grad_(False)
        opacities = self.opacities.clone().detach().requires_grad_(False)
        radiances = self.radiances.clone().detach().requires_grad_(False)
        emissives = self.emissives.clone().detach().requires_grad_(False)
        albedos = self.albedos.clone().detach().requires_grad_(False)

        if positions.isnan().sum() != 0:
            ExLog(f"{positions.isnan().sum()=}")
            breakpoint()
            exit()

        assert not positions.isnan().any()
        assert not scales.isnan().any()
        assert not quaternions.isnan().any()
        assert not opacities.isnan().any()
        assert not radiances.isnan().any()
        assert not emissives.isnan().any()
        assert not albedos.isnan().any()

        return EmissionAwareGaussians(
            count=self.count,
            positions=positions,
            scales=scales,
            quaternions=quaternions,
            opacities=opacities,
            radiances=radiances,
            emissives=emissives,
            albedos=albedos,
        )
