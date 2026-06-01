import os

from omegaconf import DictConfig

from deepforest import utilities
from deepforest.main import deepforest


def sam3_polygons(
    config: DictConfig,
    input_path: str | None = None,
    predictions_csv: str | None = None,
    output_path: str | None = None,
    root_dir: str | None = None,
    mode: str = "single",
    prompt_mode: str = "auto",
    text_prompt: str | None = None,
    model_name: str = "facebook/sam3",
    hf_token: str | None = None,
    score_threshold: float = 0.5,
    mask_threshold: float = 0.5,
    point_box_size: float = 12.0,
):
    """Convert DeepForest point/box predictions to polygons with SAM3."""
    m = deepforest(config=config)
    m.create_trainer(logger=False)

    if predictions_csv is not None:
        if root_dir is None:
            root_dir = config.validation.root_dir
        if root_dir is None:
            root_dir = os.path.dirname(predictions_csv)
        results = utilities.read_file(predictions_csv, root_dir=root_dir)
        path_for_image = input_path
    else:
        if input_path is None:
            raise ValueError("input_path is required when predictions_csv is not provided")
        path_for_image = input_path
        if mode == "single":
            results = m.predict_image(path=input_path)
        elif mode == "tile":
            results = m.predict_tile(
                path=input_path,
                patch_size=config.patch_size,
                patch_overlap=config.patch_overlap,
                iou_threshold=config.nms_thresh,
            )
        elif mode == "csv":
            if root_dir is None:
                root_dir = config.validation.root_dir
            if root_dir is None:
                root_dir = os.path.dirname(input_path)
            results = m.predict_file(csv_file=input_path, root_dir=root_dir)
            path_for_image = None
        else:
            raise ValueError(
                f"Invalid mode: {mode}. Pick one of single/tile/csv."
            )

    if results is None:
        raise ValueError("No predictions found to convert to polygons.")

    polygons = m.predict_polygons(
        results=results,
        path=path_for_image,
        root_dir=root_dir,
        text_prompt=text_prompt,
        prompt_mode=prompt_mode,
        model_name=model_name,
        hf_token=hf_token,
        score_threshold=score_threshold,
        mask_threshold=mask_threshold,
        point_box_size=point_box_size,
    )

    if output_path is not None:
        if os.path.dirname(output_path):
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
        if output_path.endswith(".shp") or output_path.endswith(".gpkg"):
            geo = utilities.image_to_geo_coordinates(polygons)
            geo.to_file(output_path)
        else:
            polygons.to_csv(output_path, index=False)

    return polygons
