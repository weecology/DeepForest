from deepforest import main
from deepforest import visualize
import os

# This script loads an image, performs tree detection using the DeepForest library, and visualizes the detected trees with bounding boxes.

# Load the image
image_path = "<path_to_image>"
image = cv2.imread(image_path)

# Create an instance of the DeepForest model
m = main.deepforest()
m.use_release()

# Perform tree detection on the image
trees = m.predict_tile(image=image, patch_size=3000, patch_overlap=0)
# Filter out low-confidence detections
trees = trees[trees.score > 0.3]
visualize.plot_results(trees, root_dir=os.path.dirname(image_path))