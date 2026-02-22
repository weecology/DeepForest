from deepforest.main import deepforest
import numpy as np

MODEL_NAMES = [
    "weecology/deepforest-bird",
    "weecology/everglades-bird-species-detector",
    "weecology/deepforest-tree",
    "weecology/deepforest-livestock",
    "weecology/cropmodel-deadtrees",
    "weecology/everglades-nest-detection",
]

for model_name in MODEL_NAMES:
    try:
        model = deepforest()
        model.load_model(model_name=model_name)
        # We need an image with something to predict, or just check the config label dict
        print(f"Model: {model_name}")
        print(f"  Numeric to label dict: {model.numeric_to_label_dict}")
    except Exception as e:
        print(f"  Error loading {model_name}: {e}")
