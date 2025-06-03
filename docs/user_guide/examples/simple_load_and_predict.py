from deepforest import main
from deepforest import visualize

# This script loads an image, performs tree detection using the DeepForest library, and visualizes the detected trees with bounding boxes.

# Create an instance of the DeepForest model
m = main.deepforest()
m.load_model(model_name="weecology/deepforest-tree")

# Perform tree detection on the image
trees = m.predict_tile(path='<path to image>', patch_size=3000, patch_overlap=0)

# Filter out low-confidence detections
trees = trees[trees.score > 0.3]
visualize.plot_results(trees)
