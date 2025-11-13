import argparse
<<<<<<< HEAD
import sys

from hydra import compose, initialize, initialize_config_dir
from omegaconf import OmegaConf

from deepforest.conf.schema import Config as StructuredConfig
from deepforest.scripts.evaluate import evaluate
from deepforest.scripts.predict import predict
from deepforest.scripts.train import train
=======
import os

from hydra import compose, initialize, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

from deepforest.conf.schema import Config as StructuredConfig
from deepforest.main import deepforest
from deepforest.visualize import plot_results


def train(config: DictConfig) -> None:
    """Train a DeepForest model with the given configuration.

    Args:
        config: Hydra configuration object containing training parameters
    """
    m = deepforest(config=config)
    m.trainer.fit(m)


def predict(
    config: DictConfig,
    input_path: str,
    output_path: str | None = None,
    plot: bool | None = False,
) -> None:
    """Run prediction for the given image, optionally saving the results to the
    provided path and optionally visualizing the results.

    Args:
        config (DictConfig): Hydra configuration.
        input_path (str): Path to the input image.
        output_path (Optional[str]): Path to save the prediction results.
        plot (Optional[bool]): Whether to plot the results.

    Returns:
        None
    """
    m = deepforest(config=config)
    res = m.predict_tile(
        path=input_path,
        patch_size=config.patch_size,
        patch_overlap=config.patch_overlap,
        iou_threshold=config.nms_thresh,
    )

    if output_path is not None:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        res.to_csv(output_path, index=False)

    if plot:
        plot_results(res)
>>>>>>> 617b2ce (Add Spotlight integration for interactive visualization)


def main():
    parser = argparse.ArgumentParser(description="DeepForest CLI")
    subparsers = parser.add_subparsers(dest="command")

    # Train subcommand
    train_parser = subparsers.add_parser(
        "train",
        help="Train a model.",
        epilog="Any remaining arguments <key>=<value> will be passed to Hydra to override the current config.",
    )
    train_parser.add_argument(
        "--disable-checkpoint",
        help="Disable model checkpointing during training",
        action="store_true",
    )
    train_parser.add_argument(
        "--comet",
        help="Enable logging to Comet ML, requires comet to be logged in.",
        action="store_true",
    )
    train_parser.add_argument(
        "--tensorboard",
        help="Enable local logging to Tensorboard",
        action="store_true",
    )
    train_parser.add_argument(
        "--trace",
        help="Enable PyTorch memory profiling.",
        action="store_true",
    )
    train_parser.add_argument(
        "--strategy",
        help="Training strategy to use (e.g., 'ddp', 'auto')",
        default="auto",
    )
    train_parser.add_argument(
        "--resume",
        help="Path to checkpoint to resume training from",
    )
    train_parser.add_argument(
        "--experiment-name",
        help="Experiment name for loggers. Overrides Comet's auto-generated name if set.",
    )
    train_parser.add_argument(
        "--tag",
        action="append",
        default=[],
        dest="tags",
        help="Tag for the experiment (can be repeated, e.g. --tag baseline --tag v2). Applied to Comet if enabled.",
    )

    # Predict subcommand
    predict_parser = subparsers.add_parser(
        "predict",
        help="Run prediction on input image or CSV file",
        epilog="Any remaining arguments <key>=<value> will be passed to Hydra to override the current config.",
    )
    predict_parser.add_argument(
        "input",
        nargs="?",
        help="Path to input image or CSV file (optional if specified in config)",
    )
    predict_parser.add_argument(
        "-o", "--output", help="Path to save prediction results CSV"
    )
    predict_parser.add_argument(
        "--plot", action="store_true", help="Visualize predictions"
    )
    predict_parser.add_argument(
        "--root-dir",
        help="Root directory containing images when input is a CSV file. Defaults to CSV directory if not specified.",
    )
    predict_parser.add_argument(
        "--mode",
        choices=["single", "tile", "csv"],
        default="single",
        help="Prediction mode: 'single' for single image, 'tile' for tiled image prediction, 'csv' for batch prediction from CSV file. Defaults to 'single'.",
    )

    # Evaluate subcommand
    evaluate_parser = subparsers.add_parser(
        "evaluate",
        help="Run evaluation on ground truth annotations. Use --predictions-csv to provide existing predictions, or omit to generate them.",
        epilog="Any remaining arguments <key>=<value> will be passed to Hydra to override the current config.",
    )
    evaluate_parser.add_argument(
        "csv_file",
        nargs="?",
        help="Path to ground truth CSV file (optional if specified in config)",
    )
    evaluate_parser.add_argument(
        "--root-dir",
        help="Root directory containing images. Defaults to CSV directory if not specified.",
    )

    evaluate_parser.add_argument(
        "--save-predictions",
        help="Path to save generated predictions CSV (only used when --predictions is not provided)",
    )
    evaluate_parser.add_argument(
        "-o", "--output", help="Path to save evaluation metrics summary CSV"
    )

    # Show config subcommand
    subparsers.add_parser("config", help="Show the current config")

    # Config options for Hydra
    parser.add_argument("--config-dir", help="Path to custom configuration directory")
    parser.add_argument(
        "--config-name", help="Name of configuration file to use", default="config"
    )

    args, overrides = parser.parse_known_args()

    if args.config_dir is not None:
        initialize_config_dir(version_base=None, config_dir=args.config_dir)
    else:
        initialize(version_base=None, config_path="pkg://deepforest.conf")

    base = OmegaConf.structured(StructuredConfig)
    cfg = compose(config_name=args.config_name, overrides=overrides)
    cfg = OmegaConf.merge(base, cfg)

    if args.command == "predict":
        predict(
            cfg,
            input_path=args.input,
            output_path=args.output,
            plot=args.plot,
            root_dir=args.root_dir,
            mode=args.mode,
        )
    elif args.command == "train":
        res = train(
            cfg,
            checkpoint=not args.disable_checkpoint,
            comet=args.comet,
            tensorboard=args.tensorboard,
            trace=args.trace,
            strategy=args.strategy,
            resume=args.resume,
            experiment_name=args.experiment_name,
            tags=args.tags,
        )

        sys.exit(0 if res else 1)

    elif args.command == "evaluate":
        evaluate(
            cfg,
            ground_truth=args.csv_file,
            root_dir=args.root_dir,
            output_path=args.output,
            save_predictions=args.save_predictions,
        )

    elif args.command == "config":
        print(OmegaConf.to_yaml(cfg, resolve=True))
<<<<<<< HEAD
<<<<<<< HEAD
    else:
        parser.print_help()
=======
    elif args.command == "gallery":
        # Gallery subcommands
        if args.gallery_cmd == "export":
            try:
                import pandas as pd
            except Exception as exc:
                raise RuntimeError(
                    "pandas is required for gallery export. Please install it in your environment."
                ) from exc

            # If demo requested, create a tiny demo dataset and image
            if args.demo:
                demo_input_dir = Path(args.out) / "demo_input"
                demo_input_dir.mkdir(parents=True, exist_ok=True)
                demo_img = demo_input_dir / "img_demo.png"
                # create a small RGB image
                Image.new("RGB", (128, 128), color=(120, 140, 160)).save(demo_img)
                df = pd.DataFrame(
                    [
                        {
                            "image_path": demo_img.name,
                            "xmin": 10,
                            "ymin": 10,
                            "xmax": 60,
                            "ymax": 60,
                            "label": "Tree",
                            "score": 0.95,
                        }
                    ]
                )
                df.root_dir = str(demo_input_dir)
            else:
                if args.input is None:
                    raise RuntimeError(
                        "Please provide an input predictions file with -i/--input"
                    )

                # read CSV or JSON depending on extension
                input_path = args.input
                if input_path.lower().endswith(".json") or input_path.lower().endswith(
                    ".jsonl"
                ):
                    df = pd.read_json(
                        input_path, lines=input_path.lower().endswith(".jsonl")
                    )
                else:
                    df = pd.read_csv(input_path)

            from deepforest.visualize import (
                export_to_gallery,
                write_gallery_html,
            )

            outdir = args.out
            export_to_gallery(
                df,
                outdir,
                root_dir=args.root_dir,
                max_crops=args.max_crops,
                sample_seed=args.sample_seed,
                sample_by_image=args.sample_by_image,
                per_image_limit=args.per_image_limit,
            )
            write_gallery_html(outdir)

            if args.start_server:
                print("Local server functionality removed - open index.html manually")
        elif args.gallery_cmd == "spotlight":
            from deepforest.visualize.spotlight_export import (
                prepare_spotlight_package,
            )

            gallery_dir = args.gallery
            outdir = args.out
            res = prepare_spotlight_package(gallery_dir, out_dir=outdir)
            print("Prepared Spotlight package:", res)
            if args.archive:
                print(
                    "Archive functionality removed - use standard tools to create archives"
                )
>>>>>>> 617b2ce (Add Spotlight integration for interactive visualization)
=======
>>>>>>> 434ed4d (Fix doc tests, patch and missing lines for Spotlight integration)


if __name__ == "__main__":
    main()
