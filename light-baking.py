from __future__ import annotations

import argparse, pathlib, re, copy

import torch


from libraries.configs import TracerConfig
from libraries.utilities import ExLog, setup_torch_and_random, UTILITIES_IO
from libraries.classes import (
    EmissionAwareGaussians,
    EAGNvsDataset,
    LearnableEmissionAwareGaussians,
)


def main(tracer_config: TracerConfig):
    # [load dataset]

    nvs_dataset: EAGNvsDataset = EAGNvsDataset.LoadBlenderTransformsSingle(
        tracer_config=tracer_config,
        transforms_json_path=pathlib.Path(
            tracer_config.NVS_DATASET_TRANSFORMS_JSON_PATH
        ),
    )

    path_tracing_output_folder = tracer_config.LIGHT_BAKING_TRAINSET_PATH_TRACED_FOLDER

    cameras_for_light_baking = copy.deepcopy(nvs_dataset.train_set_cameras)

    cam_re = re.compile(r"camera(\d+)", re.IGNORECASE)

    def camera_idx(p: pathlib.Path) -> int:
        m = cam_re.search(p.name)
        return int(m.group(1)) if m else 10**9  # files without "cameraN" go to the end

    path_tracing_renders_exr_files = sorted(
        [p for p in path_tracing_output_folder.glob("*duration*") if p.is_file()],
        key=camera_idx,
    )
    # ExLog(path_tracing_renders_exr_files)

    ExLog(f"{len(nvs_dataset.test_set_cameras)=}")

    for i_camera, camera in enumerate(cameras_for_light_baking):

        # ExLog(f"{i_camera=} {path_tracing_renders_exr_files[i_camera]=}")

        image_radiance_rgb_linear_premultiplied: torch.Tensor = (
            UTILITIES_IO.ReadExrImage(path_tracing_renders_exr_files[i_camera])
        )

        cameras_for_light_baking[
            i_camera
        ].gt_image_radiance_rgb_linear_premultiplied = (
            image_radiance_rgb_linear_premultiplied.cpu()
        )

    # [read gsply]

    gaussians: EmissionAwareGaussians = EmissionAwareGaussians.LoadPly(
        path=tracer_config.EAG_PLY_PATH
    )

    # [finetune 2D Gaussians]

    learnable_gaussians = LearnableEmissionAwareGaussians(
        gaussians=gaussians, nvs_dataset=nvs_dataset
    )
    learnable_gaussians.train(
        tracer_config=tracer_config, cameras=cameras_for_light_baking
    )


if __name__ == "__main__":
    ExLog(f"PYTHON SCRIPT START")

    setup_torch_and_random()

    parser = argparse.ArgumentParser()

    tracer_config = TracerConfig(parser=parser)

    args = parser.parse_args()

    tracer_config.extract(args=args)
    tracer_config.process()

    main(tracer_config=tracer_config)

    ExLog(f"PYTHON SCRIPT END")
