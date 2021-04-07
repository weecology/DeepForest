# DeepForest-pytorch

[![Conda package](https://github.com/weecology/DeepForest-pytorch/actions/workflows/Conda-app.yml/badge.svg)](https://github.com/weecology/DeepForest-pytorch/actions/workflows/Conda-app.yml)
[![Documentation Status](https://readthedocs.org/projects/deepforest-pytorch/badge/?version=latest)](https://deepforest-pytorch.readthedocs.io/?badge=latest)


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

```Python
from deepforest import main
m = main.deepforest()
m.create_trainer()
m.run_train()
m.evaluate(csv_file=m.config["validation"]["csv_file"], root_dir=m.config["validation"]["root_dir"])
```
[Google colab demo on model training](https://colab.research.google.com/drive/1AJUcw5dEpXeDPHd0sotAz5lpWedFYSIL#offline=true&sandboxMode=true)

## Predict a single image

```Python
from deepforest import main
csv_file = '/Users/benweinstein/Documents/DeepForest-pytorch/deepforest/data/OSBS_029.tif'
df = trained_model.predict_file(csv_file, root_dir = os.path.dirname(csv_file))
```

## Predict a large tile

```Python
prediction = trained_model.predict_tile(raster_path = raster_path,
                                        patch_size = 300,
                                        patch_overlap = 0.5,
                                        return_plot = False)
```

## Evaluate a file of annotations using intersection-over-union

```Python
csv_file = get_data("example.csv")
root_dir = os.path.dirname(csv_file)
precision, recall = m.evaluate(csv_file, root_dir, iou_threshold = 0.5)
```

# Config

DeepForest comes with a default config file (deepforest_config.yml) to control the location of training and evaluation data, the number of gpus, batch size and other hyperparameters. This file can be edited directly, or using the config dictionary after loading a deepforest object.

```
from deepforest import main
m = main.deepforest()
m.config["train"]["batch_size"] = 10
```
Config parameters are documented [here](https://deepforest-pytorch.readthedocs.io/en/latest/ConfigurationFile.html).

# Tree Detection Benchmark score

Tree detection is a central task in forest ecology and remote sensing. The Weecology Lab at the University of Florida has built a tree detection benchmark for evaluation. After building a model, you can compare it to the benchmark using the evaluate method.

```
git clone https://github.com/weecology/NeonTreeEvaluation.git
cd NeonTreeEvaluation
results = m.evaluate(csv_file = "evaluation/RGB/benchmark_annotations.csv", root_dir = "evaluation/RGB/")
results["recall"]
results["precision"]
```
