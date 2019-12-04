### NEON Benchmark

To standardize model evaluation, we have collected and published a [benchmark dataset](https://github.com/weecology/NeonTreeEvaluation) of nearly 20,000 crowns from sites in the National Ecological Observation Network. We encourage users to download and evaluate against the benchmark.

```{}
git clone https://github.com/weecology/NeonTreeEvaluation.git
cd NeonTreeEvaluation
python
```

```{python}
from deepforest import deepforest

test_model = deepforest.deepforest()
test_model.use_release()

mAP = test_model.evaluate_generator(annotations="evaluation/RGB/benchmark_annotations.csv")
print("Mean Average Precision is: {:.3f}".format(mAP))
```

```
Running network: 100% (187 of 187) |#######################################################################################################################################################| Elapsed Time: 0:09:14 Time:  0:09:14
Parsing annotations: 100% (187 of 187) |###################################################################################################################################################| Elapsed Time: 0:00:00 Time:  0:00:00
14857 instances of class Tree with average precision: 0.1892
mAP using the weighted average of precisions among classes: 0.1892
mAP: 0.1892
```
