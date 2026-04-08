import argparse

from deepforest import distributed
from deepforest.main import deepforest


def main() -> None:
    parser = argparse.ArgumentParser(description="Distributed large-tile prediction driver")
    parser.add_argument("--input-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument(
        "--model-name",
        default="weecology/everglades-bird-species-detector",
    )
    parser.add_argument("--patch-size", type=int, default=1500)
    parser.add_argument("--patch-overlap", type=float, default=0)
    parser.add_argument("--iou-threshold", type=float, default=0.15)
    parser.add_argument("--dataloader-strategy", default="window")
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--num-nodes", type=int, default=1)
    args = parser.parse_args()

    model = deepforest()
    model.load_model(model_name=args.model_name)
    model.config.accelerator = "gpu"
    model.config.devices = args.devices
    model.config.strategy = "ddp"
    model.config.num_nodes = args.num_nodes
    model.config.workers = 0
    model.create_trainer()

    results = model.predict_tile(
        path=args.input_path,
        patch_size=args.patch_size,
        patch_overlap=args.patch_overlap,
        iou_threshold=args.iou_threshold,
        dataloader_strategy=args.dataloader_strategy,
    )

    if distributed.is_global_zero(model.trainer) and results is not None:
        results.to_csv(args.output_path, index=False)


if __name__ == "__main__":
    main()
