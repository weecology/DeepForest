# How do I make the predictions better?

Give the enormous array of forest types and image acquisition environment, it is unlikely that your image will be perfectly predicted by the prebuilt model. Here are some tips to improve predictions

## Check patch size

The prebuilt model was trained on 10cm data at 400px crops. The model is sensitive to predicting to new image resolutions that differ. We have found that increasing the patch size works better on higher quality data. For example, here is a drone collected data () at the standard 400px

```
tile = model.predict_tile("/Users/ben/Desktop/test.jpg",return_plot=True,patch_overlap=0,iou_threshold=0.05,patch_size=400)
```

![](../www/example_patch400.png)

Acceptable, but not ideal.


Here is 1000 px patches.

![](../www/example_patch1000.png)


improved


## Annotate local training data

Ultimately, training a proper model with local data is the best chance at getting good performance. See DeepForest.train()
