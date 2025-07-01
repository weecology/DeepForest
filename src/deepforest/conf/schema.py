from dataclasses import dataclass, field
from typing import Optional, List, Dict, Union
from omegaconf import MISSING


@dataclass
class ModelConfig:
    name: str = "weecology/deepforest-tree"
    revision: str = "main"


@dataclass
class RetinaNetConfig:
    score_thresh: float = 0.1


@dataclass
class SchedulerParamsConfig:
    T_max: int = 10
    eta_min: float = 1e-5
    lr_lambda: str = "0.95 ** epoch"
    step_size: int = 30
    gamma: float = 0.1
    milestones: List[int] = field(default_factory=lambda: [50, 100])
    mode: str = "min"
    factor: float = 0.1
    patience: int = 10
    threshold: float = 0.0001
    threshold_mode: str = "rel"
    cooldown: int = 0
    min_lr: float = 0.0
    eps: float = 1e-8


@dataclass
class SchedulerConfig:
    type: Optional[str] = "StepLR"
    params: SchedulerParamsConfig = field(default_factory=SchedulerParamsConfig)


@dataclass
class TrainConfig:
    csv_file: Optional[str] = MISSING
    root_dir: Optional[str] = MISSING
    lr: float = 0.001
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    epochs: int = 1
    fast_dev_run: bool = False
    preload_images: bool = False


@dataclass
class ValidationConfig:
    csv_file: Optional[str] = MISSING
    root_dir: Optional[str] = MISSING
    preload_images: bool = False
    size: Optional[int] = None
    iou_threshold: float = 0.4
    val_accuracy_interval: int = 20


@dataclass
class PredictConfig:
    pin_memory: bool = False


@dataclass
class Config:
    workers: int = 0
    devices: Union[int, str] = "auto"
    accelerator: str = "auto"
    batch_size: int = 1

    architecture: str = "retinanet"
    num_classes: int = 1
    nms_thresh: float = 0.05
    model: ModelConfig = field(default_factory=ModelConfig)

    label_dict: Optional[Dict[str, int]] = None

    # Preprocessing
    path_to_raster: Optional[str] = MISSING
    patch_size: int = 400
    patch_overlap: float = 0.05
    annotations_xml: Optional[str] = MISSING
    rgb_dir: Optional[str] = MISSING
    path_to_rgb: Optional[str] = MISSING

    retinanet: RetinaNetConfig = field(default_factory=RetinaNetConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    predict: PredictConfig = field(default_factory=PredictConfig)
