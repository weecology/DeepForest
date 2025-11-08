from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    """Model configuration that defines the repository ID on HuggingFace and
    the revision (tag)."""

    name: str | None = "weecology/deepforest-tree"
    revision: str = "main"


@dataclass
class SchedulerParamsConfig:
    """Parameters used to configure the scheduler during training.

    In most cases users should not need to change these."
    """

    T_max: int = 10
    eta_min: float = 1e-5
    lr_lambda: str = "0.95 ** epoch"
    step_size: int = 30
    gamma: float = 0.1
    milestones: list[int] = field(default_factory=lambda: [50, 100])
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
    """Set the type of scheduler, by default DeepForest uses a stepped learning
    function reducing at "milestones" during training."""

    type: str | None = "StepLR"
    params: SchedulerParamsConfig = field(default_factory=SchedulerParamsConfig)


@dataclass
class TrainConfig:
    """Main training configuration. The CSV file and root directory are
    required to specify the location of the training dataset.

    The default learning rate may need to be changed for certain
    architectures, such as transformers-based models which sometimes
    prefer a lower learning rate.

    The optimizer can be "sgd" (with momentum=0.9) or "adamw". SGD is
    the default and works well for RetinaNet. AdamW is recommended for
    transformer-based models like DeformableDetr, typically with a lower
    learning rate (e.g., 1e-4 to 5e-4).

    The number of epochs should be user-specified and depends on the
    size of the dataset (e.g. how many iterations the model will train
    for and how diverse the imagery is). DeepForest uses Lightning to
    manage the training loop and you can set fast_dev_run to True for
    sanity checking.
    """

    csv_file: str | None = None
    root_dir: str | None = None
    lr: float = 0.001
    optimizer: str = "sgd"
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    epochs: int = 1
    fast_dev_run: bool = False
    preload_images: bool = False
    augmentations: list[str] | None = field(default_factory=lambda: ["HorizontalFlip"])
    log_root: str = "logs"


@dataclass
class ValidationConfig:
    """Main validation configuration. As with training data, it's required that
    you set a CSV file and root directory.

    Validation during training is important to identify if the model has
    converged or is overfitting.
    """

    csv_file: str | None = None
    root_dir: str | None = None
    preload_images: bool = False
    size: int | None = None
    iou_threshold: float = 0.4
    val_accuracy_interval: int = 20
    lr_plateau_target: str = "val_loss"
    augmentations: list[str] | None = field(default_factory=lambda: [])
    # Keypoint-specific validation (used when task="keypoint")
    pixel_distance_threshold: float = 10.0


@dataclass
class PredictConfig:
    pin_memory: bool = False


@dataclass
class CropModelConfig:
    """Configuration for the CropModel classifier training.

    This section controls the standalone crop classification module.
    """

    batch_size: int = 4
    num_workers: int = 0
    lr: float = 0.0001
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    balance_classes: bool = False
    resize: list[int] = field(default_factory=lambda: [224, 224])


@dataclass
class Config:
    """General DeepForest configuration. Some parameters here are shared
    between dataloaders, for example the batch size, accelerator and number of
    workers.

    Here we also set the architecture, which can be one of "retinanet"
    or "DeformableDetr" currently. If you modify the number of classes
    or label dict from what is loaded from the hub, it's assumed that
    you intend to fine-tune or otherwise train the model. In this case,
    the model will be adapted to fit your configuration by, for example,
    adjusting the number of classification heads.

    For most users the default setting of 1-class, "tree" should be
    sufficient.
    """

    workers: int = 0
    devices: int | str = "auto"
    accelerator: str = "auto"
    batch_size: int = 1

    task: str = "box"
    architecture: str = "retinanet"
    num_classes: int = 1
    label_dict: dict[str, int] = field(default_factory=lambda: {"Tree": 0})

    # Keypoint-specific parameters (used when task="keypoint")
    point_cost: float = 5.0
    point_loss_coefficient: float = 5.0
    point_loss_type: str = "l1"

    nms_thresh: float = 0.05
    score_thresh: float = 0.1
    model: ModelConfig = field(default_factory=ModelConfig)

    # Preprocessing
    path_to_raster: str | None = None
    patch_size: int = 400
    patch_overlap: float = 0.05
    annotations_xml: str | None = None
    rgb_dir: str | None = None
    path_to_rgb: str | None = None

    train: TrainConfig = field(default_factory=TrainConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    predict: PredictConfig = field(default_factory=PredictConfig)
    cropmodel: CropModelConfig = field(default_factory=CropModelConfig)
