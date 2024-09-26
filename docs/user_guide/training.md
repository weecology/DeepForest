# Training

The prebuilt models will always be improved by adding data from the target area. In our work, we have found that even an hour's worth of carefully chosen hand-annotation can yield enormous improvements in accuracy and precision. 5-10 epochs of fine-tuning with the prebuilt model are often adequate.

Consider an annotations.csv file in the following format

testfile_deepforest.csv

```
image_path, xmin, ymin, xmax, ymax, label
OSBS_029.jpg,256,99,288,140,Tree
OSBS_029.jpg,166,253,225,304,Tree
OSBS_029.jpg,365,2,400,27,Tree
OSBS_029.jpg,312,13,349,47,Tree
OSBS_029.jpg,365,21,400,70,Tree
OSBS_029.jpg,278,1,312,37,Tree
OSBS_029.jpg,364,204,400,246,Tree
OSBS_029.jpg,90,117,121,145,Tree
OSBS_029.jpg,115,109,150,152,Tree
OSBS_029.jpg,161,155,199,191,Tree
```

The config file specifies the path to the CSV file that we want to use when training. The images are located in the working directory by default, and a user can provide a path to a different image directory.

```python
# Example run with short training
annotations_file = get_data("testfile_deepforest.csv")

model.config["epochs"] = 1
model.config["save-snapshot"] = False
model.config["train"]["csv_file"] = annotations_file
model.config["train"]["root_dir"] = os.path.dirname(annotations_file)

model.create_trainer()
```

For debugging, its often useful to use the [fast_dev_run = True from pytorch lightning](https://pytorch-lightning.readthedocs.io/en/latest/common/trainer.html#fast-dev-run)

```
model.config["train"]["fast_dev_run"] = True
```

See [config](https://deepforest.readthedocs.io/en/latest/ConfigurationFile.html) for full set of available arguments. You can also pass any [additional](https://pytorch-lightning.readthedocs.io/en/latest/common/trainer.html) pytorch lightning argument to trainer.

To begin training, we create a pytorch-lightning trainer and call trainer.fit on the model object directly on itself.
While this might look a touch awkward, it is useful for exposing the pytorch lightning functionality.

```
model.trainer.fit(model)
```

[For more, see Google colab demo on model training](https://colab.research.google.com/drive/1gKUiocwfCvcvVfiKzAaf6voiUVL2KK_r?usp=sharing)

## Disable the progress bar

If you want to disable the progress bar while training change the `create_trainer` call to:

```python
 model.create_trainer(enable_progress_bar=False)
```

## Loggers

DeepForest logs the training loss, validation loss and class metrics (for multi-class models) during each epoch. To view the training curves, we *highly* recommend using a pytorch-lightning logger, this is the proper way of handling the many outputs during training. See [pytorch-lightning docs](https://lightning.ai/docs/pytorch/stable/extensions/logging.html) for all available loggers.

```
from deepforest import main
m = main.deepforest()
logger = <any supported pytorch lightning logger>
m.create_trainer(logger=logger)
```

### Video walkthrough of colab

<div style="position: relative; padding-bottom: 56.25%; height: 0;"><iframe src="https://www.loom.com/embed/99c55129d5a34f3dbf7053dde9c7d97e" frameborder="0" webkitallowfullscreen mozallowfullscreen allowfullscreen style="position: absolute; top: 0; left: 0; width: 100%; height: 100%;"></iframe></div>

## Reducing tile size

High resolution tiles may exceed GPU or CPU memory during training, especially when many target objecrts are present. To reduce the size of each tile, use preprocess.split_raster to divide the original tile into smaller pieces and create a corresponding annotations file.

For example, this sample data raster has size 2472, 2299 pixels.
```
"""Split raster into crops with overlaps to maintain all annotations"""
raster = get_data("2019_YELL_2_528000_4978000_image_crop2.png")
import rasterio
src = rasterio.open(raster)
/Users/benweinstein/.conda/envs/DeepForest/lib/python3.9/site-packages/rasterio/__init__.py:220: NotGeoreferencedWarning: Dataset has no geotransform, gcps, or rpcs. The identity matrix be returned.
  s = DatasetReader(path, driver=driver, sharing=sharing, **kwargs)
src.read().shape
(3, 2472, 2299)
```

With 574 trees annotations

```
annotations = utilities.read_pascal_voc(get_data("2019_YELL_2_528000_4978000_image_crop2.xml"))
annotations.shape
(574, 6)
```

```
#Write csv to file and crop
tmpdir = tempfile.gettempdir()
annotations.to_csv("{}/example.csv".format(tmpdir), index=False)
annotations_file = preprocess.split_raster(path_to_raster=raster,
                                           annotations_file="{}/example.csv".format(tmpdir),
                                           base_dir=tmpdir,
                                           patch_size=500,
                                           patch_overlap=0.25)

# Returns a 6 column pandas array
assert annotations_file.shape[1] == 6
```

Now we have crops and annotations in 500 px patches for training.

### Negative samples

To include images with no annotations from the target classes create a dummy row specifying the image_path, but set all bounding boxes to 0

```
image_path, xmin, ymin, xmax, ymax, label
myimage.png, 0,0,0,0,"Tree"
```

Excessive use of negative samples may have a negative impact on model performance, but when used sparingly, they can increase precision. 

### Model checkpoints

Pytorch lightning allows you to [save a model](https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html#checkpoint-callback) at the end of each epoch. By default this behevaior is turned off since it slows down training and quickly fills up storage. To restore model checkpointing

```
callback = ModelCheckpoint(dirpath='temp/dir',
                                 monitor='box_recall',
                                 mode="max",
                                 save_top_k=3,
                                 filename="box_recall-{epoch:02d}-{box_recall:.2f}")
model.create_trainer(logger=TensorBoardLogger(save_dir='logdir/'),
                                  callbacks=[callback])
model.trainer.fit(model)
```
### Saving and loading models

```
import tempfile
import pandas as pd
tmpdir = tempfile.TemporaryDirectory()

model.use_release()

#save the prediction dataframe after training and compare with prediction after reload checkpoint
img_path = get_data("OSBS_029.png")
model.create_trainer()
model.trainer.fit(model)
pred_after_train = model.predict_image(path = img_path)

#Create a trainer to make a checkpoint
model.trainer.save_checkpoint("{}/checkpoint.pl".format(tmpdir))

#reload the checkpoint to model object
after = main.deepforest.load_from_checkpoint("{}/checkpoint.pl".format(tmpdir))
pred_after_reload = after.predict_image(path = img_path)

assert not pred_after_train.empty
assert not pred_after_reload.empty
pd.testing.assert_frame_equal(pred_after_train,pred_after_reload)
```

---

Note that when reloading models, you should carefully inspect the model parameters, such as the score_thresh and nms_thresh. These parameters are updated during model creation and the config file is not read when loading from checkpoint!

It is best to be direct to specify after loading checkpoint. If you want to save hyperparameters, edit the deepforest_config.yml directly. This will allow the hyperparameters to be reloaded on deepforest.save_model().

---

```
after.model.score_thresh = 0.3
```

Some users have reported a pytorch lightning module error on save

In this case, just saving the torch model is an easy fix.

```
torch.save(model.model.state_dict(),model_path)
```

and restore

```
model = main.deepforest()
model.model.load_state_dict(torch.load(model_path))
```

Note that if you trained on GPU and restore on cpu, you will need the map_location argument in torch.load.


### New Augmentations

DeepForest uses the same transform for train/test, so you need to encode a switch for turning the 'augment' off.
Wrap any new augmentations like so:

```
def get_transform(augment):
    """This is the new transform"""
    if augment:
        transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            ToTensorV2()
        ], bbox_params=A.BboxParams(format='pascal_voc',label_fields=["category_ids"]))

    else:
        transform = ToTensorV2()

    return transform

m = main.deepforest(transforms=get_transform)
```

see https://albumentations.ai/docs/getting_started/bounding_boxes_augmentation/ for more options of augmentations.

**How do I make training faster?**

While it is impossible to anticipate the setup for all users, there are a few guidelines. First, a GPU-enabled processor is key. Training on a CPU can be done, but it will take much longer (100x) and is probably only done if needed. Using Google Colab can be beneficial but prone to errors. Once on the GPU, the configuration includes a "workers" argument. This connects to PyTorch's dataloader. As the number of workers increases, data is fed to the GPU in parallel. Increase the worker argument slowly, we have found that the optimal number of workers varies by system.

```
m.config["workers"] = 5
```

It is not foolproof, and occasionally 0 workers, in which data loading is run on the main thread, is optimal : https://stackoverflow.com/questions/73331758/can-ideal-num-workers-for-a-large-dataset-in-pytorch-be-0.

For large training runs, setting preload_images to True can be helpful. 

```
m.config["preload_images"] = True
```

This will load all data into GPU memory once, at the beginning of the run. This is great, but it requires you to have enough memory space to do so.
Similarly, increasing the batch size can speed up training. Like both of the options above, we have seen examples where performance (and accuracy) improves and decreases depending on batch size. Track experiment results carefully when altering batch size, since it directly [effects the speed of learning](https://www.baeldung.com/cs/learning-rate-batch-size).

```
m.config["batch_size"] = 10
```

Remember to call m.create_trainer() after updating the config dictionary.

### Avoiding **Weakly referenced objects** errors

On some devices and systems we have found an error referencing the model.trainer object that was created in m.create_trainer(). 
We welcome a reproducible issue to address this error as it appears highly variable and relates to upstream issues. It appears more common on google colab and github actions.

In most cases, this error appears when running multiple calls to model.predict or model.train. We believe this occurs because garbage collection has deleted the model.trainer object see:
https://github.com/Lightning-AI/lightning/issues/12233
https://github.com/weecology/DeepForest/issues/338

If you run into this error, users (e.g https://github.com/weecology/DeepForest/issues/443), have found that creating the trainer object within the loop can resolve this issue. 

```
for tile in tiles_to_predict:
    m.create_trainer()
    m.predict_tile(tile)
```

Usually creating this object does not cost too much computational time.

#### Training across multiple nodes on a HPC system

We have heard that this error can appear when trying to deep copy the pytorch lighnting module. The trainer object is not pickleable.
For example, on multi-gpu enviroments when trying to scale the deepforest model the entire module is copied leading to this error.
Setting the trainer object to None and directly using the pytorch object is a reasonable workaround. 

Replace

```
m = main.deepforest()
m.create_trainer()
m.trainer.fit(m)
```

with

```
m.trainer = None
from pytorch_lightning import Trainer

    trainer = Trainer(
        accelerator="gpu",
        strategy="ddp",
        devices=model.config["devices"],
        enable_checkpointing=False,
        max_epochs=model.config["train"]["epochs"],
        logger=comet_logger
    )
trainer.fit(m)
```

The added benefits of this is more control over the trainer object. 
The downside is that it doesn't align with the .config pattern where a user now has to look into the config to create the trainer. 
We are open to changing this to be the default pattern in the future and welcome input from users.
