#Profiler
from deepforest import deepforest
from deepforest import utilities
import pandas as pd

#Generate some data
annotations = utilities.xml_to_annotations("tests/data/OSBS_029.xml",rgb_dir="tests/data")
big_annotations = []
for x in range(1000):
    big_annotations.append(annotations.copy())

final_annotations = pd.concat(big_annotations)
print("data shape is {}".format(final_annotations.shape))

final_annotations.to_csv("tests/data/OSBS_029.csv",index=False, header=None)


test_model = deepforest.deepforest()
test_model.config["epochs"] = 1
test_model.config["save-snapshot"] = False
test_model.train(annotations="tests/data/OSBS_029.csv")