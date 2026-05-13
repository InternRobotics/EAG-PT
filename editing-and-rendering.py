from __future__ import annotations

import argparse, pathlib

import torch


from libraries.configs import TracerConfig
from libraries.utilities import (
    ExLog,
    setup_torch_and_random,
)
from libraries.classes import EmissionAwareGaussians, EAGNvsDataset, EAGCamera

EMISSION_THRESHOLD = 0.1


def RenderAndSave0Bounce1BouncePathTracingResults(
    edited_gaussians: EmissionAwareGaussians,
    cameras_to_render: list[EAGCamera],
    render_0_bounce: bool = True,
    render_1_bounce: bool = True,
    render_path_tracing: bool = True,
) -> None:

    if render_0_bounce:
        edited_gaussians.saveNoBounceResultsOnCameras(
            tracer_config=tracer_config,
            cameras=cameras_to_render,
            folder_path=tracer_config.OUTPUT_FOLDER_PATH / f"0-nobounce",
            is_to_save_groundtruths=True,
        )

    if render_1_bounce:
        for spp in [1024]:
            # for spp in [1]:
            edited_gaussians.saveSingleBounceResultsOnCameras(
                cameras=cameras_to_render,
                tracer_config=tracer_config,
                folder_path=tracer_config.OUTPUT_FOLDER_PATH
                / f"1-singlebounce-spp{spp}",
                spp=spp,
            )

    if render_path_tracing:
        for spp in [1024]:
            # for spp in [1]:
            edited_gaussians.savePathTracingResultsOnCameras(
                cameras=cameras_to_render,
                tracer_config=tracer_config,
                spp=spp,
                bounce_limit=7,
                folder_path=tracer_config.OUTPUT_FOLDER_PATH
                / f"2-pathtracing-spp{spp}",
            )


def main(tracer_config: TracerConfig):

    # [get cameras]

    nvs_dataset: EAGNvsDataset = EAGNvsDataset.LoadBlenderTransformsSingle(
        tracer_config=tracer_config,
        transforms_json_path=pathlib.Path(
            tracer_config.NVS_DATASET_TRANSFORMS_JSON_PATH
        ),
    )

    # [--- scene editing ---]

    # [original]
    if (
        tracer_config.I_SCENE_EDITING_SCENARIO == -1
        or tracer_config.I_SCENE_EDITING_SCENARIO == 0
    ):
        gaussians: EmissionAwareGaussians = EmissionAwareGaussians.LoadPly(
            path=tracer_config.EAG_PLY_PATH
        )

        edited_gaussians = gaussians

    # [100 - Blender-kitchen turning off all lights, and insert the lightball]
    if tracer_config.I_SCENE_EDITING_SCENARIO == 100:
        gaussians: EmissionAwareGaussians = EmissionAwareGaussians.LoadPly(
            path=tracer_config.EAG_PLY_PATH
        )
        gaussians_lightball: EmissionAwareGaussians = EmissionAwareGaussians.LoadPly(
            path="_output/Blender-assets-lightball_0-radiant/iter30000-plys/optimized-2d-gaussians.ply"
        )

        selected_indices = gaussians.emissives[:, 0] > EMISSION_THRESHOLD
        gaussians.opacities[selected_indices] = 0.0

        gaussians_lightball.positions[:, 0] += 1
        gaussians_lightball.positions[:, 1] += 0
        gaussians_lightball.positions[:, 2] += 2.25
        gaussians_with_lightball = gaussians.merge(gaussians_lightball)

        edited_gaussians = gaussians_with_lightball

    # [110 - Blender-livingroom turning off all lights, and insert the lightball]
    if tracer_config.I_SCENE_EDITING_SCENARIO == 110:
        gaussians: EmissionAwareGaussians = EmissionAwareGaussians.LoadPly(
            path=tracer_config.EAG_PLY_PATH
        )
        gaussians_lightball: EmissionAwareGaussians = EmissionAwareGaussians.LoadPly(
            path="_output/Blender-assets-lightball_0-radiant/iter30000-plys/optimized-2d-gaussians.ply"
        )

        selected_indices = gaussians.emissives[:, 0] > EMISSION_THRESHOLD
        gaussians.opacities[selected_indices] = 0.0

        gaussians_lightball.positions[:, 0] += -1.6745
        gaussians_lightball.positions[:, 1] += -3.9433
        gaussians_lightball.positions[:, 2] += 2.3537
        gaussians_with_lightball = gaussians.merge(gaussians_lightball)

        edited_gaussians = gaussians_with_lightball

    # [200 - FR-classroom change light colors]
    if tracer_config.I_SCENE_EDITING_SCENARIO == 200:
        gaussians: EmissionAwareGaussians = EmissionAwareGaussians.LoadPly(
            path=tracer_config.EAG_PLY_PATH
        )

        selected_indices_third_row = (
            (gaussians.positions[:, 2] > -4.8)
            & (gaussians.positions[:, 2] < -2.9)
            & (gaussians.emissives[:, 0] > EMISSION_THRESHOLD)
        )

        selected_indices_second_row = (
            (gaussians.positions[:, 2] > -1.3)
            & (gaussians.positions[:, 2] < 0.6)
            & (gaussians.emissives[:, 0] > EMISSION_THRESHOLD)
        )

        gaussians.radiances[selected_indices_third_row] *= torch.tensor([1.0, 1.0, 0.0])
        gaussians.radiances[selected_indices_second_row] *= torch.tensor(
            [(0x66 / 255) ** 2.2, (0xCC / 255) ** 2.2, (0xFF / 255) ** 2.2]
        )

        edited_gaussians = gaussians

    # [201 - FR-classroom turn off and on lights]
    if tracer_config.I_SCENE_EDITING_SCENARIO == 201:
        gaussians: EmissionAwareGaussians = EmissionAwareGaussians.LoadPly(
            path=tracer_config.EAG_PLY_PATH
        )

        # selected_indices = (
        #     (
        #         (gaussians.positions[:, 2] > -4.8)
        #         & (gaussians.positions[:, 2] < -2.9)
        #     )
        #     | (
        #         (gaussians.positions[:, 2] > -1.3)
        #         & (gaussians.positions[:, 2] < 0.6)
        #     )
        # ) & (gaussians.emissives[:, 0] > EMISSION_THRESHOLD)

        # [take out second row]

        indices_second_row = (
            (gaussians.positions[:, 2] > -1.3)
            & (gaussians.positions[:, 2] < 0.6)
            & (gaussians.positions[:, 1] > 1.6)
            & (gaussians.positions[:, 1] < 2.0)
            & (
                (gaussians.positions[:, 0] > 2.9) & (gaussians.positions[:, 0] < 3.8)
                | (
                    (gaussians.positions[:, 0] > -0.5)
                    & (gaussians.positions[:, 0] < 0.4)
                )
                | (
                    (gaussians.positions[:, 0] > -3.1)
                    & (gaussians.positions[:, 0] < -2.2)
                )
            )
        )
        extra_gaussians_second_row = gaussians.filter(mask_bool=indices_second_row)
        extra_gaussians_second_row.positions[:, 2] += 3.4

        # [remove first row and insert second row]

        indices_first_row = (
            (gaussians.positions[:, 2] > 2.2)
            & (gaussians.positions[:, 2] < 4.1)
            & (gaussians.positions[:, 1] > 1.6)
            & (gaussians.positions[:, 1] < 2.0)
            & (
                (gaussians.positions[:, 0] > 2.9) & (gaussians.positions[:, 0] < 3.8)
                | (
                    (gaussians.positions[:, 0] > -0.5)
                    & (gaussians.positions[:, 0] < 0.4)
                )
                | (
                    (gaussians.positions[:, 0] > -3.1)
                    & (gaussians.positions[:, 0] < -2.2)
                )
            )
        )
        indices_not_first_row = ~indices_first_row
        gaussians_except_first_row = gaussians.filter(mask_bool=indices_not_first_row)

        # merged_gaussians = gaussians_except_first_row
        merged_gaussians = gaussians_except_first_row.merge(extra_gaussians_second_row)

        # [turn off back two rows]

        selected_indices = (
            (
                (merged_gaussians.positions[:, 2] > -4.8)
                & (merged_gaussians.positions[:, 2] < -2.9)
            )
            | (
                (merged_gaussians.positions[:, 2] > -1.3)
                & (merged_gaussians.positions[:, 2] < 0.6)
            )
        ) & (merged_gaussians.emissives[:, 0] > EMISSION_THRESHOLD)
        merged_gaussians.radiances[selected_indices] = 0.0
        merged_gaussians.emissives[selected_indices] = 0.0

        edited_gaussians = merged_gaussians

    # [202 - FR-classroom duplicate chairs]
    if tracer_config.I_SCENE_EDITING_SCENARIO == 202:
        gaussians: EmissionAwareGaussians = EmissionAwareGaussians.LoadPly(
            path=tracer_config.EAG_PLY_PATH
        )

        selected_indices_floor = (
            # z
            (gaussians.positions[:, 2] > -4.1)
            & (gaussians.positions[:, 2] < -3.0)
            # x
            & (gaussians.positions[:, 0] > 1.9)
            & (gaussians.positions[:, 0] < 2.7)
            # y
            & (gaussians.positions[:, 1] > -2.0)
            & (gaussians.positions[:, 1] < -1.0)
        )

        gaussians_duplicated_chair_left = gaussians.filter(
            mask_bool=selected_indices_floor
        )
        gaussians_duplicated_chair_left.positions[:, 0] -= 1.2
        gaussians_duplicated_chair_left.positions[:, 2] += 0.2

        selected_indices_chair_right = (
            # z
            (gaussians.positions[:, 2] > -3.9)
            & (gaussians.positions[:, 2] < -2.8)
            # x
            & (gaussians.positions[:, 0] > -0.8)
            & (gaussians.positions[:, 0] < 0.3)
            # y
            & (gaussians.positions[:, 1] > -2.0)
            & (gaussians.positions[:, 1] < -1.0)
        )

        gaussians_duplicated_chair_right = gaussians.filter(
            mask_bool=selected_indices_chair_right
        )
        gaussians_duplicated_chair_right.positions[:, 0] -= 1.2
        gaussians_duplicated_chair_right.positions[:, 2] += 0.2

        # merged_gaussians = gaussians_except_first_row
        merged_gaussians = gaussians.merge(gaussians_duplicated_chair_left).merge(
            gaussians_duplicated_chair_right
        )

        edited_gaussians = merged_gaussians

    # [203 - FR-classroom Teaser]
    if tracer_config.I_SCENE_EDITING_SCENARIO == 203:

        gaussians: EmissionAwareGaussians = EmissionAwareGaussians.LoadPly(
            path=tracer_config.EAG_PLY_PATH
        )

        # [1: take out second row]

        indices_second_row = (
            (gaussians.positions[:, 2] > -1.3)
            & (gaussians.positions[:, 2] < 0.6)
            & (gaussians.positions[:, 1] > 1.6)
            & (gaussians.positions[:, 1] < 2.0)
            & (
                ((gaussians.positions[:, 0] > 2.9) & (gaussians.positions[:, 0] < 3.8))
                | (
                    (gaussians.positions[:, 0] > -0.5)
                    & (gaussians.positions[:, 0] < 0.4)
                )
                | (
                    (gaussians.positions[:, 0] > -3.1)
                    & (gaussians.positions[:, 0] < -2.2)
                )
            )
        )
        extra_gaussians_second_row = gaussians.filter(indices_second_row)
        extra_gaussians_second_row.positions[:, 2] += 3.4
        extra_gaussians_second_row.radiances *= 1.5

        # [remove first row and insert second row]

        indices_first_row = (
            (gaussians.positions[:, 2] > 2.2)
            & (gaussians.positions[:, 2] < 4.1)
            & (gaussians.positions[:, 1] > 1.6)
            & (gaussians.positions[:, 1] < 2.0)
            & (
                ((gaussians.positions[:, 0] > 2.9) & (gaussians.positions[:, 0] < 3.8))
                | (
                    (gaussians.positions[:, 0] > -0.5)
                    & (gaussians.positions[:, 0] < 0.4)
                )
                | (
                    (gaussians.positions[:, 0] > -3.1)
                    & (gaussians.positions[:, 0] < -2.2)
                )
            )
        )
        gaussians.opacities[indices_first_row] = 0.0

        # merged_gaussians = gaussians_except_first_row
        merged_gaussians = gaussians.merge(extra_gaussians_second_row)

        # [1: turn off the last row]

        selected_indices_third_row = (
            (merged_gaussians.positions[:, 2] > -4.8)
            & (merged_gaussians.positions[:, 2] < -2.9)
            & (merged_gaussians.emissives[:, 0] > EMISSION_THRESHOLD)
        )
        merged_gaussians.emissives[selected_indices_third_row] = 0.0
        merged_gaussians.radiances[selected_indices_third_row] = 0.0

        # [1: change the color of second row]

        selected_indices_second_row = (
            (merged_gaussians.positions[:, 2] > -1.3)
            & (merged_gaussians.positions[:, 2] < 0.6)
            & (merged_gaussians.emissives[:, 0] > EMISSION_THRESHOLD)
        )

        # merged_gaussians.radiances[selected_indices_second_row] *= torch.tensor([1.0, 1.0, 0.0])  # weird yellow
        merged_gaussians.radiances[selected_indices_second_row] *= (
            torch.tensor(
                [(0x66 / 255) ** 2.2, (0xCC / 255) ** 2.2, (0xFF / 255) ** 2.2]
            )
            * 1.0
        )  # lower the radiance

        # [2: insert diffuse ball]

        gaussians_lightball: EmissionAwareGaussians = EmissionAwareGaussians.LoadPly(
            path="_output/Blender-assets-lightball_0-radiant/iter30000-plys/optimized-2d-gaussians.ply"
        )
        gaussians_lightball.positions[:, 1] -= 1.72  # move down
        gaussians_lightball.positions[:, 2] += 0.75  # move front

        gaussians_lightball.emissives[...] = 0.0
        gaussians_lightball.radiances[...] = 0.3
        # ExLog(f"{gaussians_lightball.albedos.mean()=}")  # 0.2
        gaussians_lightball.albedos[...] = 0.6

        merged_gaussians = merged_gaussians.merge(gaussians_lightball)

        # [3: chair color]

        # selected_indices_chair_back_left = (
        #     # z
        #     (merged_gaussians.positions[:, 2] > -4.1)
        #     & (merged_gaussians.positions[:, 2] < -3.0)
        #     # x
        #     & (merged_gaussians.positions[:, 0] > 1.9)
        #     & (merged_gaussians.positions[:, 0] < 2.7)
        #     # y
        #     & (merged_gaussians.positions[:, 1] > -2.0)
        #     & (merged_gaussians.positions[:, 1] < -1.0)
        # )
        # left_chair_gaussians = merged_gaussians.filter(selected_indices_chair_back_left)

        selected_indices_chair_back_left = (
            # z
            (merged_gaussians.positions[:, 2] > -1.8)
            & (merged_gaussians.positions[:, 2] < -0.8)
            # x
            & (merged_gaussians.positions[:, 0] > 1.9)
            & (merged_gaussians.positions[:, 0] < 3.0)
            # y
            & (merged_gaussians.positions[:, 1] > -2.0)
            & (merged_gaussians.positions[:, 1] < -1.0)
        )
        left_chair_gaussians = merged_gaussians.filter(selected_indices_chair_back_left)

        # move the position
        left_chair_gaussians.positions[:, 0] -= 2.0  # move right
        left_chair_gaussians.positions[:, 2] -= 0.1  # move back

        # change to purple
        tmp = left_chair_gaussians.radiances[:, 2]
        left_chair_gaussians.radiances[:, 2] = left_chair_gaussians.radiances[:, 0]
        left_chair_gaussians.radiances[:, 0] = tmp
        tmp = left_chair_gaussians.albedos[:, 2]
        left_chair_gaussians.albedos[:, 2] = left_chair_gaussians.albedos[:, 0]
        left_chair_gaussians.albedos[:, 0] = tmp

        merged_gaussians = merged_gaussians.merge(left_chair_gaussians)

        # [4: ]

        DEBUG_GAP = 0.6
        DEBUG_RADIANCE_MULTIPLIER = 0.20
        DEBUG_X_POSITION = 4.4
        DEBUG_Y_POSITION = 1.3
        DEBUG_Z_POSITION = -2.9
        DEBUG_Z_OFFSET = 0.4

        COLOR_YELLOW = [0.8235, 0.7647, 0.5137]
        COLOR_ORANGE = [0.7765, 0.6078, 0.4784]
        COLOR_VIOLET = [0.738, 0.574, 0.809]
        COLOR_GREEN = [0.5608, 0.7176, 0.4431]
        COLOR_RED = [0.6510, 0.4275, 0.4196]

        # [4:1]

        gaussians_lightball: EmissionAwareGaussians = EmissionAwareGaussians.LoadPly(
            path="_output/Blender-assets-lightball_0-radiant/iter30000-plys/optimized-2d-gaussians.ply"
        )

        # scale 1/2
        gaussians_lightball.positions[...] *= 0.6
        gaussians_lightball.scales[...] *= 0.6
        # move
        gaussians_lightball.positions[...] += torch.tensor(
            [
                DEBUG_X_POSITION,
                DEBUG_Y_POSITION - DEBUG_GAP * 0,
                DEBUG_Z_POSITION + DEBUG_Z_OFFSET * 0,
            ]
        )[None, :]
        # dimmer
        gaussians_lightball.radiances[...] *= (
            torch.tensor(COLOR_ORANGE)[None, :] ** 2.2 * DEBUG_RADIANCE_MULTIPLIER
        )

        merged_gaussians = merged_gaussians.merge(gaussians_lightball)

        # [4:2]

        gaussians_lightball: EmissionAwareGaussians = EmissionAwareGaussians.LoadPly(
            path="_output/Blender-assets-lightball_0-radiant/iter30000-plys/optimized-2d-gaussians.ply"
        )

        # scale 1/2
        gaussians_lightball.positions[...] *= 0.6
        gaussians_lightball.scales[...] *= 0.6
        # move
        gaussians_lightball.positions[...] += torch.tensor(
            [
                DEBUG_X_POSITION,
                DEBUG_Y_POSITION - DEBUG_GAP * 1,
                DEBUG_Z_POSITION + DEBUG_Z_OFFSET * 1,
            ]
        )[None, :]
        # dimmer
        gaussians_lightball.radiances[...] *= (
            torch.tensor(COLOR_YELLOW)[None, :] ** 2.2 * DEBUG_RADIANCE_MULTIPLIER
        )

        merged_gaussians = merged_gaussians.merge(gaussians_lightball)

        # [4:3]

        gaussians_lightball: EmissionAwareGaussians = EmissionAwareGaussians.LoadPly(
            path="_output/Blender-assets-lightball_0-radiant/iter30000-plys/optimized-2d-gaussians.ply"
        )

        # scale 1/2
        gaussians_lightball.positions[...] *= 0.6
        gaussians_lightball.scales[...] *= 0.6
        # move
        gaussians_lightball.positions[...] += torch.tensor(
            [
                DEBUG_X_POSITION,
                DEBUG_Y_POSITION - DEBUG_GAP * 2,
                DEBUG_Z_POSITION + DEBUG_Z_OFFSET * 2,
            ]
        )[None, :]
        # dimmer
        gaussians_lightball.radiances[...] *= (
            torch.tensor(COLOR_GREEN)[None, :] ** 2.2 * DEBUG_RADIANCE_MULTIPLIER
        )

        merged_gaussians = merged_gaussians.merge(gaussians_lightball)

        # [4:4]

        gaussians_lightball: EmissionAwareGaussians = EmissionAwareGaussians.LoadPly(
            path="_output/Blender-assets-lightball_0-radiant/iter30000-plys/optimized-2d-gaussians.ply"
        )

        # scale 1/2
        gaussians_lightball.positions[...] *= 0.6
        gaussians_lightball.scales[...] *= 0.6
        # move
        gaussians_lightball.positions[...] += torch.tensor(
            [
                DEBUG_X_POSITION,
                DEBUG_Y_POSITION - DEBUG_GAP * 3,
                DEBUG_Z_POSITION + DEBUG_Z_OFFSET * 1,
            ]
        )[None, :]
        # dimmer
        gaussians_lightball.radiances[...] *= (
            torch.tensor(COLOR_RED)[None, :] ** 2.2 * DEBUG_RADIANCE_MULTIPLIER
        )

        merged_gaussians = merged_gaussians.merge(gaussians_lightball)

        # [4:5]

        gaussians_lightball: EmissionAwareGaussians = EmissionAwareGaussians.LoadPly(
            path="_output/Blender-assets-lightball_0-radiant/iter30000-plys/optimized-2d-gaussians.ply"
        )

        # scale 1/2
        gaussians_lightball.positions[...] *= 0.6
        gaussians_lightball.scales[...] *= 0.6
        # move
        gaussians_lightball.positions[...] += torch.tensor(
            [
                DEBUG_X_POSITION,
                DEBUG_Y_POSITION - DEBUG_GAP * 4,
                DEBUG_Z_POSITION + DEBUG_Z_OFFSET * 0,
            ]
        )[None, :]
        # dimmer
        gaussians_lightball.radiances[...] *= (
            torch.tensor(COLOR_VIOLET)[None, :] ** 2.2 * DEBUG_RADIANCE_MULTIPLIER
        )

        merged_gaussians = merged_gaussians.merge(gaussians_lightball)

        # [5: import E-furnishedroom]

        gaussians_furnishedroom: EmissionAwareGaussians = (
            EmissionAwareGaussians.LoadPly(
                path="_output/EFT-furnishedroom_1-diffuse/iter400-plys/optimized-2d-gaussians_iter400.ply"
            )
        )

        selected_indices_lamp = (
            # z
            (gaussians_furnishedroom.positions[:, 2] > -2.85)
            & (gaussians_furnishedroom.positions[:, 2] < -2.55)
            # x
            & (gaussians_furnishedroom.positions[:, 0] > 1.95)
            & (gaussians_furnishedroom.positions[:, 0] < 2.20)
            # y
            & (gaussians_furnishedroom.positions[:, 1] > 0.5)
            & (gaussians_furnishedroom.positions[:, 1] < 1.4)
        )

        gaussians_furnishedroom_light = gaussians_furnishedroom.filter(
            mask_bool=selected_indices_lamp
        )

        re_center_mean = gaussians_furnishedroom_light.positions.mean(
            dim=0, keepdim=True
        )
        gaussians_furnishedroom_light.positions[...] -= re_center_mean
        # ExLog(f"{gaussians_furnishedroom_light.positions.mean(dim=0, keepdim=True)=}")
        # exit()
        gaussians_furnishedroom_light.positions[:, 1] -= 0.915
        gaussians_furnishedroom_light.positions[:, 2] += 3.0
        gaussians_furnishedroom_light.positions[:, 0] -= 0.5

        merged_gaussians = merged_gaussians.merge(gaussians_furnishedroom_light)

        # [render]

        edited_gaussians = merged_gaussians

    # [301 - only keep front lights]
    if tracer_config.I_SCENE_EDITING_SCENARIO == 301:

        gaussians: EmissionAwareGaussians = EmissionAwareGaussians.LoadPly(
            path=tracer_config.EAG_PLY_PATH
        )

        is_light = (
            (gaussians.positions[:, 0] > -3.0)
            & (gaussians.positions[:, 0] < 2.7)
            & (gaussians.emissives[:, 0] > EMISSION_THRESHOLD)
        )

        selected_indices_1 = (
            (gaussians.positions[:, 2] > 2.7)
            & (gaussians.positions[:, 2] < 3.3)
            & is_light
        )
        selected_indices_2 = (
            (gaussians.positions[:, 2] > 1.2)
            & (gaussians.positions[:, 2] < 1.7)
            & is_light
        )
        selected_indices_3 = (
            (gaussians.positions[:, 2] > -0.4)
            & (gaussians.positions[:, 2] < 0.1)
            & is_light
        )
        selected_indices_4 = (
            (gaussians.positions[:, 2] > -2.0)
            & (gaussians.positions[:, 2] < -1.6)
            & is_light
        )

        # gaussians.radiances[selected_indices_3] = 0.0
        # gaussians.emissives[selected_indices_3] = 0.0
        # gaussians.radiances[selected_indices_4] = 0.0
        # gaussians.emissives[selected_indices_4] = 0.0

        gaussians.opacities[selected_indices_3] = 0.0
        gaussians.opacities[selected_indices_4] = 0.0

        edited_gaussians = gaussians

    # [302 - only keep back lights]
    if tracer_config.I_SCENE_EDITING_SCENARIO == 302:

        gaussians: EmissionAwareGaussians = EmissionAwareGaussians.LoadPly(
            path=tracer_config.EAG_PLY_PATH
        )

        is_light = (
            (gaussians.positions[:, 0] > -3.0)
            & (gaussians.positions[:, 0] < 2.7)
            & (gaussians.emissives[:, 0] > EMISSION_THRESHOLD)
        )

        selected_indices_1 = (
            (gaussians.positions[:, 2] > 2.7)
            & (gaussians.positions[:, 2] < 3.3)
            & is_light
        )
        selected_indices_2 = (
            (gaussians.positions[:, 2] > 1.2)
            & (gaussians.positions[:, 2] < 1.7)
            & is_light
        )
        selected_indices_3 = (
            (gaussians.positions[:, 2] > -0.4)
            & (gaussians.positions[:, 2] < 0.1)
            & is_light
        )
        selected_indices_4 = (
            (gaussians.positions[:, 2] > -2.0)
            & (gaussians.positions[:, 2] < -1.6)
            & is_light
        )

        # gaussians.radiances[selected_indices_1] = 0.0
        # gaussians.emissives[selected_indices_1] = 0.0
        # gaussians.radiances[selected_indices_2] = 0.0
        # gaussians.emissives[selected_indices_2] = 0.0

        gaussians.opacities[selected_indices_1] = 0.0
        gaussians.opacities[selected_indices_2] = 0.0

        edited_gaussians = gaussians

    # [E-kitchen insert plane]
    if tracer_config.I_SCENE_EDITING_SCENARIO == 400:

        gaussians: EmissionAwareGaussians = EmissionAwareGaussians.LoadPly(
            path=tracer_config.EAG_PLY_PATH
        )

        # [plane]

        gaussians_plane: EmissionAwareGaussians = EmissionAwareGaussians.LoadPly(
            path="_output/Blender-assets-plane_0-radiant/iter30000-plys/optimized-2d-gaussians.ply"
        )

        SCALE_FACTOR = 6.0
        gaussians_plane.scales *= SCALE_FACTOR
        gaussians_plane.positions *= SCALE_FACTOR

        gaussians_plane.albedos[...] = 1.0
        gaussians_plane.positions[:, 1] += 2.0
        gaussians_plane.positions[:, 2] += -1.5

        gaussians = gaussians.merge(gaussians_plane)

        # [plane]

        gaussians_plane: EmissionAwareGaussians = EmissionAwareGaussians.LoadPly(
            path="_output/Blender-assets-plane_0-radiant/iter30000-plys/optimized-2d-gaussians.ply"
        )

        SCALE_FACTOR = 6.0
        gaussians_plane.scales *= SCALE_FACTOR
        gaussians_plane.positions *= SCALE_FACTOR

        gaussians_plane.albedos[...] = 1.0
        gaussians_plane.positions[:, 1] += 2.0
        gaussians_plane.positions[:, 2] += 7.0

        gaussians = gaussians.merge(gaussians_plane)

        edited_gaussians = gaussians

    # [E-kitchen turn off light and insert light ball]
    if tracer_config.I_SCENE_EDITING_SCENARIO == 401:

        gaussians: EmissionAwareGaussians = EmissionAwareGaussians.LoadPly(
            path=tracer_config.EAG_PLY_PATH
        )

        # [turn off light]

        indices_room_ceiling = (
            # y
            (gaussians.positions[:, 1] > 2.2)
            & (gaussians.emissives[:, 0] > 0.1)
        )
        gaussians.emissives[indices_room_ceiling] = 0.0
        gaussians.radiances[indices_room_ceiling] = 0.0

        # [lightball]

        gaussians_lightball: EmissionAwareGaussians = EmissionAwareGaussians.LoadPly(
            path="_output/Blender-assets-lightball_0-radiant/iter30000-plys/optimized-2d-gaussians.ply"
        )

        # gaussians_lightball.scales *= 0.8
        # gaussians_lightball.positions *= 0.8

        gaussians_lightball.positions[:, 1] += 1.5
        gaussians_lightball.positions[:, 0] += 2.0
        gaussians_lightball.positions[:, 2] += 3.0

        gaussians = gaussians.merge(gaussians_lightball)

        # [lightball]

        gaussians_lightball: EmissionAwareGaussians = EmissionAwareGaussians.LoadPly(
            path="_output/Blender-assets-lightball_0-radiant/iter30000-plys/optimized-2d-gaussians.ply"
        )

        gaussians_lightball.scales *= 0.8
        gaussians_lightball.positions *= 0.8

        gaussians_lightball.positions[:, 1] += 1.2
        gaussians_lightball.positions[:, 0] += -2.5
        gaussians_lightball.positions[:, 2] += 2.5

        gaussians_lightball.radiances *= 0.6

        gaussians = gaussians.merge(gaussians_lightball)

        edited_gaussians = gaussians

    # [E-furnishedroom rainbow ceiling light]
    if tracer_config.I_SCENE_EDITING_SCENARIO == 410:

        gaussians: EmissionAwareGaussians = EmissionAwareGaussians.LoadPly(
            path=tracer_config.EAG_PLY_PATH
        )

        indices_room_ceiling = (
            # y
            (gaussians.positions[:, 1] > 2.5)
            & (gaussians.emissives[:, 0] > 0.1)
        )

        indices_light_column_1 = (gaussians.positions[:, 0] > -3.3) & (
            gaussians.positions[:, 0] < -1.6
        )
        indices_light_column_2 = (gaussians.positions[:, 0] > -1.6) & (
            gaussians.positions[:, 0] < 0.3
        )
        indices_light_column_3 = (gaussians.positions[:, 0] > 0.3) & (
            gaussians.positions[:, 0] < 2.0
        )

        indices_light_row_1 = (gaussians.positions[:, 2] > -0.8) & (
            gaussians.positions[:, 2] < 0.0
        )
        indices_light_row_2 = (gaussians.positions[:, 2] > -1.8) & (
            gaussians.positions[:, 2] < -0.8
        )
        indices_light_row_3 = (gaussians.positions[:, 2] > -2.6) & (
            gaussians.positions[:, 2] < -1.8
        )
        indices_light_row_4 = (gaussians.positions[:, 2] > -3.6) & (
            gaussians.positions[:, 2] < -2.6
        )
        indices_light_row_5 = (gaussians.positions[:, 2] > -4.6) & (
            gaussians.positions[:, 2] < -3.6
        )

        indices_light_1 = indices_room_ceiling & (
            (gaussians.positions[:, 0] > -2.8)
            & (gaussians.positions[:, 0] < -1.0)
            & (gaussians.positions[:, 2] > 0.5)
            & (gaussians.positions[:, 2] < 1.5)
        )
        indices_light_2 = (
            indices_room_ceiling & indices_light_column_1 & indices_light_row_1
        )
        indices_light_3 = (
            indices_room_ceiling & indices_light_column_1 & indices_light_row_3
        )
        indices_light_4 = (
            indices_room_ceiling & indices_light_column_1 & indices_light_row_5
        )

        indices_light_5 = (
            indices_room_ceiling & indices_light_column_2 & indices_light_row_2
        )
        indices_light_6 = (
            indices_room_ceiling & indices_light_column_2 & indices_light_row_4
        )

        indices_light_7 = (
            indices_room_ceiling & indices_light_column_3 & indices_light_row_1
        )
        indices_light_8 = (
            indices_room_ceiling & indices_light_column_3 & indices_light_row_3
        )
        indices_light_9 = (
            indices_room_ceiling & indices_light_column_3 & indices_light_row_5
        )

        RADIANCE_SCALE = 1.0

        COLOR_YELLOW = [0.8235, 0.7647, 0.5137]
        COLOR_ORANGE = [0.7765, 0.6078, 0.4784]
        COLOR_VIOLET = [0.738, 0.574, 0.809]
        COLOR_GREEN = [0.5608, 0.7176, 0.4431]
        COLOR_RED = [0.6510, 0.4275, 0.4196]

        gaussians.radiances[indices_light_1] *= (
            torch.tensor(COLOR_YELLOW)[None, :] * RADIANCE_SCALE
        )
        gaussians.radiances[indices_light_2] *= (
            torch.tensor(COLOR_ORANGE)[None, :] * RADIANCE_SCALE
        )
        gaussians.radiances[indices_light_3] *= (
            torch.tensor(COLOR_VIOLET)[None, :] * RADIANCE_SCALE
        )
        gaussians.radiances[indices_light_4] *= (
            torch.tensor(COLOR_GREEN)[None, :] * RADIANCE_SCALE
        )
        gaussians.radiances[indices_light_5] *= (
            torch.tensor(COLOR_RED)[None, :] * RADIANCE_SCALE
        )

        gaussians.radiances[indices_light_6] *= (
            torch.tensor(COLOR_YELLOW)[None, :] * RADIANCE_SCALE
        )
        gaussians.radiances[indices_light_7] *= (
            torch.tensor(COLOR_ORANGE)[None, :] * RADIANCE_SCALE
        )
        gaussians.radiances[indices_light_8] *= (
            torch.tensor(COLOR_VIOLET)[None, :] * RADIANCE_SCALE
        )
        gaussians.radiances[indices_light_9] *= (
            torch.tensor(COLOR_GREEN)[None, :] * RADIANCE_SCALE
        )

        gaussians.emissives[indices_light_5] *= 0.0
        gaussians.emissives[indices_light_6] *= 0.0

        gaussians.radiances[indices_light_5] *= 0.0
        gaussians.radiances[indices_light_6] *= 0.0

        edited_gaussians = gaussians

    # [E-furnishedroom turn off light and insert light ball]
    if tracer_config.I_SCENE_EDITING_SCENARIO == 411:

        gaussians: EmissionAwareGaussians = EmissionAwareGaussians.LoadPly(
            path=tracer_config.EAG_PLY_PATH
        )

        indices_room_ceiling = (
            # y
            (gaussians.positions[:, 1] > 2.5)
            & (gaussians.emissives[:, 0] > 0.1)
        )
        gaussians.emissives[indices_room_ceiling] = 0.0
        gaussians.radiances[indices_room_ceiling] = 0.0

        gaussians_lightball: EmissionAwareGaussians = EmissionAwareGaussians.LoadPly(
            path="_output/Blender-assets-lightball_0-radiant/iter30000-plys/optimized-2d-gaussians.ply"
        )

        gaussians_lightball.scales *= 0.8
        gaussians_lightball.positions *= 0.8

        gaussians_lightball.positions[:, 1] += 0.9
        gaussians_lightball.positions[:, 0] += -1.1
        gaussians_lightball.positions[:, 2] += -2.7

        gaussians = gaussians.merge(gaussians_lightball)

        edited_gaussians = gaussians

    # [E-kitchen counter to E-emptyroom]
    if tracer_config.I_SCENE_EDITING_SCENARIO == 420:

        gaussians: EmissionAwareGaussians = EmissionAwareGaussians.LoadPly(
            path=tracer_config.EAG_PLY_PATH
        )

        gaussians_E_kitchen: EmissionAwareGaussians = EmissionAwareGaussians.LoadPly(
            path="_output/EFT-kitchen_1-diffuse/iter400-plys/optimized-2d-gaussians_iter400.ply"
        )

        z_____min = -0.075
        indices_E_kitchen_counter = (
            # z
            (gaussians_E_kitchen.positions[:, 2] > 0.5)
            & (gaussians_E_kitchen.positions[:, 2] < 5.2)
            # x
            & (gaussians_E_kitchen.positions[:, 0] > -0.2)
            & (gaussians_E_kitchen.positions[:, 0] < 1.2)
            # y
            & (gaussians_E_kitchen.positions[:, 1] > z_____min)
            & (gaussians_E_kitchen.positions[:, 1] < 1.6)
        )
        gaussians_E_kitchen_counter = gaussians_E_kitchen.filter(
            indices_E_kitchen_counter
        )
        maxxx = gaussians_E_kitchen_counter.positions.max(dim=0, keepdim=True)[0][0]
        minnn = gaussians_E_kitchen_counter.positions.min(dim=0, keepdim=True)[0][0]
        # ExLog(f"{maxxx=} {minnn=}"); exit()
        gaussians_E_kitchen_counter.positions[:, 0] -= (maxxx[0] + minnn[0]) / 2
        gaussians_E_kitchen_counter.positions[:, 2] -= (maxxx[2] + minnn[2]) / 2
        gaussians_E_kitchen_counter.positions[:, 1] += -z_____min

        OFFSET_0 = -1.3
        OFFSET_2 = -1.9
        gaussians_E_kitchen_counter.positions[:, 0] += OFFSET_0
        gaussians_E_kitchen_counter.positions[:, 2] += OFFSET_2

        gaussians = gaussians.merge(gaussians_E_kitchen_counter)

        edited_gaussians = gaussians

    # [E-emptyroom change wall to green]
    if tracer_config.I_SCENE_EDITING_SCENARIO == 421:

        gaussians: EmissionAwareGaussians = EmissionAwareGaussians.LoadPly(
            path=tracer_config.EAG_PLY_PATH
        )

        indices_emptyroom_wall = (
            # z
            (gaussians.positions[:, 2] < -4.7)
            # x
            & (gaussians.positions[:, 0] > -3.5)
        )
        gaussians.albedos[indices_emptyroom_wall] *= torch.tensor([0.2, 0.8, 0.2])[
            None, :
        ]

        gaussians.radiances[indices_emptyroom_wall] *= torch.tensor([0.2, 0.8, 0.2])[
            None, :
        ]

        edited_gaussians = gaussians

    # [--- scene editing ---]

    edited_gaussians.savePly(tracer_config.OUTPUT_FOLDER_PATH / "edited.ply")

    if not tracer_config.RENDER_FOR_LIGHT_BAKING:
        # [common]
        RenderAndSave0Bounce1BouncePathTracingResults(
            edited_gaussians=edited_gaussians,
            cameras_to_render=nvs_dataset.test_set_cameras,
            # cameras_to_render=nvs_dataset.train_set_cameras[63-1],  # for FR-classroom Teaser
        )
    else:
        # [for trainset light baking]
        RenderAndSave0Bounce1BouncePathTracingResults(
            edited_gaussians=edited_gaussians,
            cameras_to_render=nvs_dataset.train_set_cameras,
            render_1_bounce=False,
        )


if __name__ == "__main__":
    ExLog(f"PYTHON SCRIPT START")

    print()

    setup_torch_and_random()
    parser = argparse.ArgumentParser()
    tracer_config = TracerConfig(parser=parser)
    args = parser.parse_args()
    tracer_config.extract(args=args)
    tracer_config.process()

    print()

    main(tracer_config=tracer_config)

    print()

    ExLog(f"PYTHON SCRIPT END")
