import pathlib, datetime, zoneinfo, argparse

from libraries.utilities import ExLog


class BaseConfig:
    # https://stackoverflow.com/a/52403318/14298786
    def StrToBool(value: str):
        if isinstance(value, bool):
            return value
        if value.lower() in {"false", "f", "0", "no", "n"}:
            return False
        elif value.lower() in {"true", "t", "1", "yes", "y"}:
            return True
        raise ValueError(f"{value} is not a valid boolean value")

    def __init__(self, parser: argparse.ArgumentParser):
        ExLog(f"default config: {vars(self)}")
        for key, default_value in vars(self).items():
            if type(default_value) == bool:
                parser.add_argument(
                    "--" + key,
                    type=BaseConfig.StrToBool,
                    default=default_value,
                )
            else:
                parser.add_argument(
                    "--" + key,
                    type=type(default_value),
                    default=default_value,
                )

    def extract(self, args: argparse.Namespace):
        for key, new_value in vars(args).items():
            if key in vars(self):
                old_value = getattr(self, key)
                if old_value != new_value:
                    ExLog(f"modified argument: {key} ({old_value} -> {new_value})")
                setattr(self, key, new_value)
        ExLog(f"modified config: {vars(self)}")


class TracerConfig(BaseConfig):
    def __init__(self, parser: argparse.ArgumentParser = None):

        # [NVS Dataset]

        self.NVS_DATASET_PATH: pathlib.Path = pathlib.Path(
            f"./ENTER_YOUR_NVS_DATASET_PATH"
        )

        # [real datasets / scenes]

        self.DATASET_IS_SYNTHETIC: bool = False

        # [training]

        # self.ITERATION: int = 6400
        self.ITERATION: int = 30000

        # [learning rate]

        self.LEARNING_RATE_MULTIPLE: float = 1.0

        # https://github.com/graphdeco-inria/gaussian-splatting/blob/main/arguments/__init__.py
        # original 3DGS: self.position_lr_init = 0.00016
        #                self.position_lr_final = 0.0000016
        #                self.position_lr_delay_mult = 0.01
        self.LEARNING_RATE_POSITION: float = 0.000016
        self.LEARNING_RATE_SCALE: float = 0.005
        self.LEARNING_RATE_QUATERNION: float = 0.001
        self.LEARNING_RATE_OPACITY: float = 0.05
        self.LEARNING_RATE_RADIANCE: float = 0.01

        self.LEARNING_RATE_EMISSIVE: float = 0.05

        # [optimize albedo]

        self.ITERATION_OPTIMIZE_ALBEDO: int = 400

        self.N_SPP_OPTIMIZE_ALBEDO: int = 16

        self.LEARNING_RATE_ALBEDO: float = 0.05

        # [training]

        self.SAVE_ITER0_RENDERS: bool = True

        # [loss]

        self.LOSS_OVERALL_RADIANCE: float = 1.0

        self.LOSS_LAMBDA_DSSIM: float = 0.2
        self.LOSS_ALPHA_CHANNEL_SUPERVISION: float = (
            0.1  # TODO can add this later, but should not start from iter0
        )

        self.LOSS_EMISSIVE_MASK_SUPERVISION: float = 0.1

        self.LOSS_NORMAL_SUPERVISION: float = 0.00
        self.NORMAL_SUPERVISION_START_FROM_ITER: int = 3_000

        self.LOSS_DEPTH_SUPERVISION: float = 0.00
        self.DEPTH_SUPERVISION_START_FROM_ITER: int = 3_000

        self.LOSS_NORMAL_CONSISTENCY: float = 0.00
        self.NORMAL_CONSISTENCY_START_FROM_ITER: int = 3_000

        # [EAG-PT ply]

        self.EAG_PLY_PATH: pathlib.Path = pathlib.Path(f"./ENTER_YOUR_EAG_PLY_PATH")

        # [Scene Editing]

        self.I_SCENE_EDITING_SCENARIO: int = 0

        # [Output]

        self.OUTPUT_FOLDER_SUFFIX: str = ""

        # [Light Baking]

        self.LIGHT_BAKING_TRAINSET_PATH_TRACED_FOLDER: pathlib.Path = pathlib.Path(
            f"./ENTER_YOUR_FOLDER"
        )

        self.RENDER_FOR_LIGHT_BAKING: bool = False

        self.RENDER_TRAIN_SET_AND_BAKE_LIGHT_INTO_2D_GAUSSIANS_0_BOUNCE: bool = False

        # [TSDF Mesh Export]

        # for general indoor scenes
        self.TSDF_MESH_EXPORT_DEPTH_TRUNC: float = 8.0
        self.TSDF_MESH_EXPORT_MESH_RESOLUTION: float = 256
        self.TSDF_MESH_EXPORT_NUM_CLUSTERS: int = 50

        # # for lightball
        # self.TSDF_MESH_EXPORT_DEPTH_TRUNC: float = 2.0
        # self.TSDF_MESH_EXPORT_MESH_RESOLUTION: float = 256
        # self.TSDF_MESH_EXPORT_NUM_CLUSTERS: int = 1

        # [Path Tracing cameras selection]

        self.SELECTED_CAMERAS_TO_PATH_TRACING_A: int = -1
        self.SELECTED_CAMERAS_TO_PATH_TRACING_B: int = -1

        # [ConfigBase]

        super().__init__(parser=parser)

    def process(self):
        # [time]

        self.TIME_PREFIX_STR = datetime.datetime.now(
            tz=zoneinfo.ZoneInfo("Asia/Shanghai")
        ).strftime("%y%m%d-%H%M%S")

        # [downsampling for EFT dataset]

        if "EFT" in str(self.NVS_DATASET_PATH):
            self.DOWN_SAMPLE_SCALE_WHEN_LOADING_DATA: float = 1024.0 / 540.0
        else:
            self.DOWN_SAMPLE_SCALE_WHEN_LOADING_DATA: float = 1.0

        # [mine]

        self.NVS_DATASET_TRANSFORMS_JSON_PATH: pathlib.Path = (
            self.NVS_DATASET_PATH / "transforms.json"
        )

        self.NVS_DATASET_TRANSFORMS_TRAIN_JSON_PATH: pathlib.Path = (
            self.NVS_DATASET_PATH / "transforms_train.json"
        )
        self.NVS_DATASET_TRANSFORMS_TEST_JSON_PATH: pathlib.Path = (
            self.NVS_DATASET_PATH / "transforms_test.json"
        )

        # [iteration]

        self.ITERATIONS_TO_EVALUATE: list[int] = [
            # 400,
            # 1000,
            # 3000,
            # 15000,
            self.ITERATION,
        ]
        self.ITERATIONS_TO_SAVE: list[int] = [
            # 400,
            # 1000,
            # 3000,
            # 15000,
            self.ITERATION,
        ]

        self.ITERATIONS_TO_SAVE_OPTIMIZE_ALBEDO: list[int] = [
            # 400,
            # 800,
            # 1600,
            self.ITERATION_OPTIMIZE_ALBEDO,
        ]
        self.ITERATIONS_TO_EVALUATE_OPTIMIZE_ALBEDO: list[int] = [
            # 400,
            # 800,
            # 1600,
            self.ITERATION_OPTIMIZE_ALBEDO,
        ]

        # [output]

        self.ADD_OUTPUT_PREFIX: bool = False

        if self.ADD_OUTPUT_PREFIX:
            self.OUTPUT_FOLDER_PATH: pathlib.Path = pathlib.Path(
                f"./_output/{self.TIME_PREFIX_STR}{f'_{self.OUTPUT_FOLDER_SUFFIX}' if self.OUTPUT_FOLDER_SUFFIX != '' else ''}/"
            )
        else:
            self.OUTPUT_FOLDER_PATH: pathlib.Path = pathlib.Path(
                f"./_output/{f'{self.OUTPUT_FOLDER_SUFFIX}' if self.OUTPUT_FOLDER_SUFFIX != '' else ''}/"
            )

        self.OUTPUT_FOLDER_PATH.mkdir(parents=True, exist_ok=True)

        # [config]

        ExLog(f"processed config: {vars(self)=}")
