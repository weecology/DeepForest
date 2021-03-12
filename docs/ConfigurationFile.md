# Config

Deepforest-pytorch uses a config.yml to control hyperparameters related to model training and evaluation. This allows all the relevant parameters to live in one location and be easily changed when exploring new models.

Deepforest-pytorch comes with a sample config file, deepforest_config.yml. Edit this file to change settings while developing models.

```
# Config file for DeepForest-pytorch module

#cpu workers for data loaders
#Dataloaders
workers: 1
gpus: 
distributed_backend:
batch_size: 1

#Non-max supression of overlapping predictions
nms_thresh: 0.05
score_thresh: 0.1

train:

    csv_file:
    root_dir:
    
    #Optomizer  initial learning rate
    lr: 0.001

    #Print loss every n epochs
    print_freq: 1
    epochs: 1
    #Useful debugging flag in pytorch lightning, set to True to get a single batch of training to test settings.
    fast_dev_run: False
    
validation:
    #callback args
    csv_file: 
    root_dir:
    #Intersection over union evaluation
    iou_threshold: 0.4
```

## Dataloaders

### workers
Number of workers to perform asynchronous data generation during model training. Corresponds to num_workers in pytorch base 
class https://pytorch.org/docs/stable/data.html. To turn off asynchronous data generation set workers = 0.

### gpus
The number of gpus to use during model training. To run on cpu set to 0. Deepforest-pytorch has been tested on up to 8 gpu and follows a pytorch lightning module, which means it can inherent any of the scaling functionality from this library, including TPU support.
https://pytorch-lightning.readthedocs.io/en/stable/advanced/multi_gpu.html?highlight=multi%20gpu

### distributed_backend
Data parallelization strategy from https://pytorch-lightning.readthedocs.io/en/stable/advanced/multi_gpu.html?highlight=multi%20gpu. 


