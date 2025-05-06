import argparse
import os
import sys
from typing import Optional

from hydra import compose, initialize
from omegaconf import DictConfig, OmegaConf

from deepforest.main import deepforest
from deepforest.visualize import plot_results


def predict(config: DictConfig,
            input_path: str,
            output_path: Optional[str] = None,
            plot: Optional[bool] = False) -> None:
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
    m.load_model(model_name=config.model.name, revision=config.model.revision)
    res = m.predict_tile(path=input_path,
                         patch_size=config.patch_size,
                         patch_overlap=config.patch_overlap,
                         iou_threshold=config.nms_thresh)

    if output_path is not None:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        res.to_csv(output_path, index=False)

    if plot:
        plot_results(res)


def main():
    parser = argparse.ArgumentParser(description='DeepForest prediction utility')
    parser.add_argument("input", help="Path to raster", nargs="?")
    parser.add_argument("-o", "--output", help="Path to prediction results")
    parser.add_argument("--plot", help="Plot results", action="store_true")
    parser.add_argument("--show-config",
                        action="store_true",
                        help="Show available config overrides and exit")
    args, overrides = parser.parse_known_args()

    with initialize(version_base=None, config_path="pkg://deepforest.conf"):

        if args.show_config:
            print("\Configuration overview:\n")
            cfg = compose(config_name="config")
            print(OmegaConf.to_yaml(cfg, resolve=True))
            sys.exit(0)
        elif args.input is None:
            parser.error("the following argument is required: input")
        else:
            cfg = compose(config_name="config", overrides=overrides)

    predict(cfg, input_path=args.input, output_path=args.output, plot=args.plot)


if __name__ == "__main__":
    main()
