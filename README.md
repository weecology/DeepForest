# DeepForest-pytorch

[![Conda package](https://github.com/weecology/DeepForest-pytorch/actions/workflows/Conda-app.yml/badge.svg)](https://github.com/weecology/DeepForest-pytorch/actions/workflows/Conda-app.yml)

A pytorch implementation of the DeepForest model for individual tree crown detection in RGB images. DeepForest is a python package for training and predicting individual tree crowns from airborne RGB imagery. DeepForest comes with a prebuilt model trained on data from the National Ecological Observatory Network. Users can extend this model by annotating and training custom models starting from the prebuilt model.

<sub> DeepForest es un paquete de python para la predicción de coronas de árboles individuales basada en modelos entrenados con imágenes remotas RVA ( RGB, por sus siglas en inglés). DeepForest viene con un modelo entrenado con datos proveídos por la Red Nacional de Observatorios Ecológicos (NEON, por sus siglas en inglés). Los usuarios pueden ampliar este modelo pre-construido por anotación de etiquetas y entrenamiento con datos locales. La documentación de DeepForest está escrita en inglés, sin embargo, agradeceríamos contribuciones con fin de hacerla accesible en otros idiomas.  <sub>

 <sub> DeepForest(PyTorch版本)是一个Python软件包，它可以被用来训练以及预测机载RGB图像中的单个树冠。DeepForest内部带有一个基于国家生态观测站网络(NEON : National Ecological Observatory Network)数据训练的预训练模型。在此模型基础上，用户可以注释新的数据然后训练自己的模型。DeepForest的文档是用英文编写的，如果您有兴趣为翻译文档做出贡献。欢迎与我们团队联系。<sub>

## Motivation

The original DeepForest repo is written in tensorflow and can be found on pypi, conda and source (https://github.com/Weecology/DeepForest). After https://github.com/fizyr/keras-retinanet was deprecated, it became obvious that the shelf life of models that depend on tensorflow 1.0 was limited. The machine learning community is moving more towards pytorch, where many new models can be found. 

# Installation

Compiled wheels have been made for linux, osx and windows

```
#Install DeepForest-pytorch
pip install deepforest-pytorch
```

# Usage

## Train a model

```
from deepforest import main
m = main.deepforest()
m.create_trainer()
m.run_train()
m.evaluate(csv_file=m.config["validation"]["csv_file"], root_dir=m.config["validation"]["root_dir"])
```

## Predict a single image

```
from deepforest import main
csv_file = '/Users/benweinstein/Documents/DeepForest-pytorch/deepforest/data/OSBS_029.tif'
df = trained_model.predict_file(csv_file, root_dir = os.path.dirname(csv_file))
```

## Predict a large tile

```
prediction = trained_model.predict_tile(raster_path = raster_path,
                                        patch_size = 300,
                                        patch_overlap = 0.5,
                                        return_plot = False)
```

## Evaluate a file of annotations using intersection-over-union

```
csv_file = get_data("example.csv")
root_dir = os.path.dirname(csv_file)
precision, recall = m.evaluate(csv_file, root_dir, iou_threshold = 0.5)
```

# Status

v0.0.2 will be the first stable release pending the following milestones.

1. Train a deepforest-pytorch release model
2. Update docs and tutorials to reflect new workflow
3. Clean builds on travis and azure dev ops for pypi builds.


# Config


# Benchmark score

Tree detection is a central task in forest ecology and remote sensing. The Weecology Lab at the University of Florida has built a tree detection benchmark for evaluation.

```
git clone 

```


# Integration with pytorch-lightning

DeepForest-pytorch will follow the pytorch-lightning (https://www.pytorchlightning.ai/) philosophy for maximum reproducibility. DeepForest objects are now lightning modules that can access any of the excellent functionalities from that framework.

```
from deepforest import main
from pytorch_lightning.callbacks import Callback
ls

m = main.deepforest()

csv_file = get_data("example.csv") 
root_dir = os.path.dirname(csv_file)
train_ds = m.load_dataset(csv_file, root_dir=root_dir)
  
class MyPrintingCallback(Callback):

    def on_init_start(self, trainer):
        print('Starting to init trainer!')

    def on_init_end(self, trainer):
        print('trainer is init now')

    def on_train_end(self, trainer, pl_module):
        print('do something when training ends')
  
trainer = Trainer(callbacks=[MyPrintingCallback()])
  
trainer = Trainer(fast_dev_run=True)
trainer.fit(m, train_ds) 
```

```
Starting to init trainer!
GPU available: False, used: False
TPU available: None, using: 0 TPU cores
trainer is init now
GPU available: False, used: False
TPU available: None, using: 0 TPU cores
Running in fast_dev_run mode: will run a full train, val and test loop using 1 batch(es)

  | Name     | Type      | Params
---------------------------------------
0 | backbone | RetinaNet | 34.0 M
---------------------------------------
33.8 M    Trainable params
222 K     Non-trainable params
34.0 M    Total params

Training: 0it [00:00, ?it/s]
Training:   0%|          | 0/1 [00:00<?, ?it/s]
Epoch 0:   0%|          | 0/1 [00:00<?, ?it/s]
Epoch 0: 100%|██████████| 1/1 [00:16<00:00, 16.90s/it]
Epoch 0: 100%|██████████| 1/1 [00:16<00:00, 16.90s/it, loss=3.15, v_num=31]
Epoch 0: 100%|██████████| 1/1 [00:22<00:00, 22.33s/it, loss=3.15, v_num=31]
Reading config file: deepforest_config.yml
```
