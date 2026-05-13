from __future__ import annotations

import argparse, pathlib, time

import torch, plyfile, numpy as np


from libraries.configs import TracerConfig
from libraries.utilities import ExLog, setup_torch_and_random, UTILITIES_COLOUR
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

    if True:

        # [initialize from point cloud]

        points: plyfile.PlyElement = plyfile.PlyData.read(
            tracer_config.NVS_DATASET_PATH / "points3d.ply"
        )["vertex"]
        ExLog(
            f"Read {points.count} points from {tracer_config.NVS_DATASET_PATH / 'points3d.ply'}."
        )

        initial_positions: np.ndarray = np.column_stack(
            (
                points["x"],
                points["y"],
                points["z"],
            )
        )
        initial_rgbs: np.ndarray = (
            np.column_stack((points["red"], points["green"], points["blue"])).astype(
                np.float32
            )
            / 255.0
        )

        initial_gaussians = EmissionAwareGaussians(
            count=points.count,
            positions=torch.tensor(initial_positions, dtype=torch.float32),
            scales=torch.ones((points.count, 2)) * 0.01,
            quaternions=EmissionAwareGaussians.Normalize(torch.rand(points.count, 4)),
            opacities=torch.ones((points.count, 1)) * 0.1,
            radiances=UTILITIES_COLOUR.SrgbToLinear(
                torch.tensor(initial_rgbs, dtype=torch.float32)
            ),
            emissives=torch.ones((points.count, 1)) * 0.1,
            albedos=torch.ones((points.count, 3)) * 0.2,
        )

    else:

        # [load from previous checkpoint]

        initial_gaussians = EmissionAwareGaussians.LoadPly(path=tracer_config.EAG_PLY_PATH)

    ExLog(f"{initial_gaussians.count=}")

    if True:

        # [train 2D Gaussians]

        learnable_gaussians = LearnableEmissionAwareGaussians(
            gaussians=initial_gaussians, nvs_dataset=nvs_dataset
        )

        learnable_gaussians.train(
            tracer_config=tracer_config, cameras=nvs_dataset.train_set_cameras
        )


if __name__ == "__main__":
    ExLog(f"PYTHON SCRIPT START")

    setup_torch_and_random()

    parser = argparse.ArgumentParser()

    tracer_config = TracerConfig(parser=parser)

    args = parser.parse_args()

    tracer_config.extract(args=args)
    tracer_config.process()

    time_start = time.perf_counter()
    main(tracer_config=tracer_config)
    time_end = time.perf_counter()
    main_duration = time_end - time_start
    with open(tracer_config.OUTPUT_FOLDER_PATH / "main-duration.py", "w") as f:
        f.write(f"main_duration_seconds={main_duration}\n")
        f.write(f"main_duration_minutes={main_duration/60}\n")
    ExLog(f"Saved main_duration at {tracer_config.OUTPUT_FOLDER_PATH / 'main-duration.py'}")

    ExLog(f"PYTHON SCRIPT END")
