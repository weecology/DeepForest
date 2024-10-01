# Multi-species models

DeepForest allows training on multiple species annotations.
When creating a deepforest model object, pass the designed number of classes and a label dictionary that maps each numeric class to a character label. The number of classes can be either be specified in the config, or using config_args during creation.

``` python
m = main.deepforest(config_args={"num_classes":2},label_dict={"Alive":0,"Dead":1})
```

It is often, but not always, useful to start with a prebuilt model when trying to identify multiple species. This helps the model focus on learning the multiple classes and not waste data and time re-learning bounding boxes. To load the backbone and box prediction portions of the release model, but create a classification model for more than one species.
Here is an example using the alive/dead tree data stored in the package, but the same logic applies to the bird detector. 

``` python
# Initialize new Deepforest model ( the model that you will train ) with your classes
m = main.deepforest(config_args={"num_classes":2}, label_dict={"Alive":0,"Dead":1})

# Inatialize Deepforest model ( the model that you will modify its regression head ) 
deepforest_release_model = main.deepforest()
deepforest_release_model.use_release() # or use_bird_release

# Extract single class backbone that will have useful features for multi-class classification
m.model.backbone.load_state_dict(deepforest_release_model.model.backbone.state_dict())

# load regression head in the new model
m.model.head.regression_head.load_state_dict(deepforest_release_model.model.head.regression_head.state_dict())

m.config["train"]["csv_file"] = get_data("testfile_multi.csv") 
m.config["train"]["root_dir"] = os.path.dirname(get_data("testfile_multi.csv"))
m.config["train"]["fast_dev_run"] = True
m.config["batch_size"] = 2
    
m.config["validation"]["csv_file"] = get_data("testfile_multi.csv") 
m.config["validation"]["root_dir"] = os.path.dirname(get_data("testfile_multi.csv"))
m.config["validation"]["val_accuracy_interval"] = 1

m.create_trainer()
m.trainer.fit(m)
```

* For more on loading with state_dict: [Pytorch Docs](https://pytorch.org/tutorials/beginner/saving_loading_models.html#save-load-state-dict-recommended)