# Evaluation wrapper
from src import evaluate
from deepforest import main

# Current backbone
m = main.deepforest()
m.use_release()
current_backbone = evaluate.wrapper(m)
print(current_backbone)
current_backbone.to_csv("results/current_backbone_eval.csv")