#Training module
# Developed to mimic https://github.com/pytorch/vision/blob/master/references/detection/train.py

import math
import torch
import torch.optim as optim
from deepforest import training_utils as utils

def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1. / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = utils.warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)

    for images, targets in metric_logger.log_every(data_loader, print_freq, header):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)

        losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    return metric_logger

def run(train_ds, model, config, debug=False):
    """Train a Deepforest model in pytorch
    Args:
        train_ds: a pytorch dataset, see main.load_dataset
        model: a deepforest model see main.create() or main.load_model()
        config: a deepforest config object
        debug: used for tests, to keep training loop short. Take 1 batch from the data to train
    """
    
    #put model to device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")    
    model = model.to(device)
        
    #set configs
    # Observe that all parameters are being optimized
    optimizer = optim.SGD(model.parameters(), lr=config["lr"], momentum=0.9)
    
    #Randomly sample each epoch
    train_sampler = torch.utils.data.RandomSampler(train_ds)
    train_batch_sampler = torch.utils.data.BatchSampler(
        train_sampler, config["batch_size"], drop_last=True)
    
    if debug:
        data_loader_train = torch.utils.data.DataLoader(
            train_ds, sampler=torch.utils.data.sampler.SubsetRandomSampler([1]), num_workers=config["workers"],
            collate_fn=utils.collate_fn)        
    else:
        data_loader_train = torch.utils.data.DataLoader(
            train_ds, batch_sampler=train_batch_sampler, num_workers=config["workers"],
            collate_fn=utils.collate_fn)        
    
    num_epochs = config["epochs"]
    for epoch in range(num_epochs):
        train_one_epoch(model, optimizer, data_loader_train, device, epoch, print_freq=config["print_freq"])
    
    return model

