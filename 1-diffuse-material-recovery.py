from __future__ import annotations

import argparse, pathlib, time


import torch

from libraries.configs import TracerConfig
from libraries.utilities import (
    ExLog,
    setup_torch_and_random,
)
from libraries.classes import (
    EmissionAwareGaussians,
    EAGNvsDataset,
    LearnableEmissionAwareGaussians,
)


def main(tracer_config: TracerConfig):
    # [load checkpoint]

    gaussians: EmissionAwareGaussians = EmissionAwareGaussians.LoadPly(
        path=tracer_config.EAG_PLY_PATH
    )

    # [use normalized trained radiances as initial albedos]

    gaussians.albedos = torch.nn.functional.normalize(
        gaussians.radiances.clone().clamp(min=0.01, max=0.99)
    )

    ExLog(
        f"{gaussians.radiances.mean()=} {gaussians.radiances.min()=} {gaussians.radiances.max()=}"
    )
    ExLog(
        f"{gaussians.albedos.mean()=} {gaussians.albedos.min()=} {gaussians.albedos.max()=}"
    )

    # [get camera]

    nvs_dataset: EAGNvsDataset = EAGNvsDataset.LoadBlenderTransformsSingle(
        tracer_config=tracer_config,
        transforms_json_path=pathlib.Path(
            tracer_config.NVS_DATASET_TRANSFORMS_JSON_PATH
        ),
    )

    # [use LearnableGaussians to optimize diffuse albedos]

    learnable_gaussians = LearnableEmissionAwareGaussians(
        gaussians=gaussians, nvs_dataset=nvs_dataset
    )
    learnable_gaussians.optimizeAlbedosUsingSingleBounceIntoRadianceCache(
        tracer_config=tracer_config
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

    time_start = time.perf_counter()
    main(tracer_config=tracer_config)
    time_end = time.perf_counter()
    main_duration = time_end - time_start
    with open(tracer_config.OUTPUT_FOLDER_PATH / "main-duration.py", "w") as f:
        f.write(f"main_duration_seconds={main_duration}\n")
        f.write(f"main_duration_minutes={main_duration/60}\n")
    ExLog(
        f"Saved main_duration at {tracer_config.OUTPUT_FOLDER_PATH / 'main-duration.py'}"
    )

    print()

    ExLog(f"PYTHON SCRIPT END")
