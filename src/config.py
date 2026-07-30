from dataclasses import dataclass
import dataclasses
import yaml


@dataclass
class ModelConfig:
    name_or_path: str
    dtype: str = "float16"
    device_map: str = "auto"


@dataclass
class DataConfig:
    source: str
    pairs_path: str
    n_samples: int
    train_test_split: float
    split_seed: int


@dataclass
class RepEConfig:
    rep_token: int
    direction_method: str
    n_difference: int
    batch_size: int


@dataclass
class OutputConfig:
    vectors_dir: str
    figures_dir: str
    accuracy_threshold: float = 0.7


@dataclass
class Stage1Config:
    seed: int
    run_name: str
    model: ModelConfig
    data: DataConfig
    repe: RepEConfig
    output: OutputConfig


def load_stage1_config(path: str) -> Stage1Config:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return Stage1Config(
        seed=raw["seed"],
        run_name=raw["run_name"],
        model=ModelConfig(**raw["model"]),
        data=DataConfig(**raw["data"]),
        repe=RepEConfig(**raw["repe"]),
        output=OutputConfig(**raw["output"]),
    )


@dataclass
class ClipConfig:
    backbone: str = "ViT-B-32"
    pretrained: str = "openai"


@dataclass
class Stage1VectorConfig:
    directions_path: str  # negation_directions_all_layers.npy from a stage1 run
    layer: int             # which layer's direction to use (stage1's best layer)


@dataclass
class ImageEvalConfig:
    manifest_path: str
    n_samples: int
    seed: int


@dataclass
class Stage2Config:
    seed: int
    run_name: str
    model: ModelConfig
    clip: ClipConfig
    data: DataConfig
    image_eval: ImageEvalConfig
    stage1_vectors: Stage1VectorConfig
    output: OutputConfig


def load_stage2_config(path: str) -> Stage2Config:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return Stage2Config(
        seed=raw["seed"],
        run_name=raw["run_name"],
        model=ModelConfig(**raw["model"]),
        clip=ClipConfig(**raw["clip"]),
        data=DataConfig(**raw["data"]),
        image_eval=ImageEvalConfig(**raw["image_eval"]),
        stage1_vectors=Stage1VectorConfig(**raw["stage1_vectors"]),
        output=OutputConfig(**raw["output"]),
    )


@dataclass
class VectorPathsConfig:
    stage2_run_dir: str
    transferred_dir_file: str
    native_dir_file: str


@dataclass
class Stage2CorrectionConfig:
    seed: int
    run_name: str
    clip: ClipConfig
    image_eval: ImageEvalConfig
    vectors: VectorPathsConfig
    alphas: list
    output: OutputConfig


def load_stage2_correction_config(path: str) -> Stage2CorrectionConfig:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return Stage2CorrectionConfig(
        seed=raw["seed"],
        run_name=raw["run_name"],
        clip=ClipConfig(**raw["clip"]),
        image_eval=ImageEvalConfig(**raw["image_eval"]),
        vectors=VectorPathsConfig(**raw["vectors"]),
        alphas=raw["alphas"],
        output=OutputConfig(**raw["output"]),
    )


@dataclass
class LayerSweepConfig:
    directions_path: str
    candidate_layers: list


@dataclass
class Stage2LayerSweepConfig:
    seed: int
    run_name: str
    model: ModelConfig
    clip: ClipConfig
    data: DataConfig
    layer_sweep: LayerSweepConfig
    output: OutputConfig


def load_stage2_layer_sweep_config(path: str) -> Stage2LayerSweepConfig:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return Stage2LayerSweepConfig(
        seed=raw["seed"],
        run_name=raw["run_name"],
        model=ModelConfig(**raw["model"]),
        clip=ClipConfig(**raw["clip"]),
        data=DataConfig(**raw["data"]),
        layer_sweep=LayerSweepConfig(**raw["layer_sweep"]),
        output=OutputConfig(**raw["output"]),
    )


def save_resolved_config(cfg, path: str):
    def to_dict(obj):
        if dataclasses.is_dataclass(obj):
            return {k: to_dict(v) for k, v in dataclasses.asdict(obj).items()}
        return obj

    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(to_dict(cfg), f, sort_keys=False)