# Evaluation wrapper
import warnings
warnings.simplefilter(action='ignore')
from src import evaluate
from deepforest import main

# Current backbone
m = main.deepforest()
m.config["batch_size"] = 24
m.config["workers"] = 0
m.use_release()

# Box recall
current_backbone = evaluate.box_wrapper(m)
print(current_backbone)
current_backbone.to_csv("results/current_backbone_box_eval.csv")

# Point recall
current_backbone = evaluate.point_wrapper(m)
print(current_backbone)
current_backbone.to_csv("results/current_backbone_point_eval.csv")
