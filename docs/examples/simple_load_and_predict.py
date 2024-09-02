from deepforest import main
from deepforest import visualize
import supervision as sv
import cv2
import matplotlib.pyplot as plt

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

# Convert the tree detections to Supervision format for visualization
sv_detections = visualize.convert_to_sv_format(trees)

# Create a bounding box annotator
bounding_box_annotator = sv.BoxAnnotator()

# Annotate the image with bounding boxes
annotated_frame = bounding_box_annotator.annotate(
    scene=image,
    detections=sv_detections
)

# Display the annotated image using Matplotlib
plt.imshow(annotated_frame)
plt.axis('off')  # Hide axes for a cleaner look
plt.show()
