# Multi-class Modeling

DeepForest is primarily designed for individual tree detection. The training data, the evaluation routine, and the visualization tools are all geared towards this aim. However, the underlying engine (keras-retinanet) is agnostic to the number of classes. This means that users who want to experience with multi-class models may do so at their own risk. It is not known how well RGB data can be used to predict tree classes, but recent research suggests it is possible. For those interested in multiple classes a few words of intuition. Please note that these are all hypotheses at this point, much more work is needed to validate the best way forward.

1. Label every tree in your image, regardless of classes

Deepforest learns on a per image basis. Therefore if you have trees in an image that belong to a class, but are not labeled, you are confusing the network and will get poor performance. Field-collected data are often incomplete, with some trees labeled and other trees unlabeled. If so, consider (#3) and try to use a cascaded network strategy. If you are labeling the image by hand,  we anticipate a two class model of "Target Species" and "Other" will perform better than simply labeling the target species. It is worth trying both approaches, but I would anticipate it being more effective and in line with existing model weights.

2. Multi-class losses will take longer to train.

Finetuning of the prebuilt model often takes only a few epochs (5-10). I anticipate that a multi-class model will take many epochs (>100) to both learn the spatial position of the tree, and its target class. If you are monitoring the losses in comet or looking at the deepforest.plot_curves(), I anticipate that the bounding box regression loss will drop significantly before the class loss.

3. Consider a cascaded network to separately predict classes and trees.

An alternative strategy is to first use DeepForest is delineate trees, and then feed these crops into a 2nd neural network for classification. This may be an easier way of dealing with incomplete labels.

A couple notes on multi-class, if you trained a multi-class model and need to reload the model, make sure to reload the classes file, or else all objects will be labeled (tree)[https://github.com/weecology/DeepForest/issues/150]

```
from deepforest import deepforest
from deepforest import utilities
m = deepforest.deepforest("/orange/ewhite/everglades/Zooniverse/predictions/20210211_072221.h5")
m.classes_file = utilities.create_classes("/orange/ewhite/everglades/Zooniverse/parsed_images/test.csv")
m.read_classes()
```

Also note there is likely some integration errors with the comet dashboard, I recommend not using comet for multi-species models, as there are too many assumptions for single tree species.

