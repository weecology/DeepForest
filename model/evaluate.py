# Evaluation wrapper
from src import evaluate
from deepforest import main

# Current backbone
m = main.deepforest()
m.config["workers"] = 0
m.use_release()
current_backbone = evaluate.wrapper(m)
print(current_backbone)