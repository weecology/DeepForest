# Extending the deepforest module

DeepForest-pytorch is a pytorch lightning module. This means that any of the class methods can be extended or overwritten. See pytorch lightning's [extensive documentation](https://pytorch-lightning.readthedocs.io/en/latest/starter/introduction_guide.html).

Here is a quick example. Suppose you want to log the training metrics, which is not done by default. We could overwrite the [training_step](https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html?highlight=training_step#training-loop). See an example of pytorch lightning logging [here](https://pytorch-lightning.readthedocs.io/en/latest/common/loggers.html).

We can subclass the main deepforest-pytorch module, use super() to init all the normal class methods, and then just overwrite the method we would like to change.

```
#Overwrite default training logs and lr
class mymodule(main.deepforest):
    def __init__(self):
        super().__init__()
    
    def training_step(self, batch, batch_idx):
        """Train on a loaded dataset
        """
        path, images, targets = batch
    
        loss_dict = self.model.forward(images, targets)
    
        # sum of regression and classification loss
        losses = sum([loss for loss in loss_dict.values()])
        # Log loss
        for key, value in loss_dict.items():
            self.log("train_{}".format(key), value, on_epoch=True)
            
        return losses
```

Now when we call this module, it has the changed training_step, but still has all the other methods we would like, such as downloading the release model.

```
m = mymodule()
m.use_release() 

```
In this way we can make additions and changes without needing to edit the deepforest-pytorch source.