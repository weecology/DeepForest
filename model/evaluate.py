# Evaluation wrapper
import warnings
warnings.simplefilter(action='ignore')
from src import evaluate
from deepforest import main

# Current backbone
m = main.deepforest()
m.config["batch_size"] = 2
m.config["workers"] = 10
m.use_release()
current_backbone = evaluate.wrapper(m)
print(current_backbone)
current_backbone.to_csv("results/current_backbone_eval.csv")