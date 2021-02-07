# DeepForest-pytorch

[![Build Status](https://travis-ci.org/weecology/DeepForest-pytorch.svg?branch=master)](https://travis-ci.org/weecology/DeepForest-pytorch)

A pytorch implementation of the DeepForest model for individual tree crown detection in RGB images. DeepForest is a python package for training and predicting individual tree crowns from airborne RGB imagery. DeepForest comes with a prebuilt model trained on data from the National Ecological Observatory Network. Users can extend this model by annotating and training custom models starting from the prebuilt model.

DeepForest es un paquete de python para la predicción de coronas de árboles individuales basada en modelos entrenados con imágenes remotas RVA ( RGB, por sus siglas en inglés). DeepForest viene con un modelo entrenado con datos proveídos por la Red Nacional de Observatorios Ecológicos (NEON, por sus siglas en inglés). Los usuarios pueden ampliar este modelo pre-construido por anotación de etiquetas y entrenamiento con datos locales. La documentación de DeepForest está escrita en inglés, sin embargo, agradeceríamos contribuciones con fin de hacerla accesible en otros idiomas.

DeepForest(PyTorch版本)是一个Python软件包，它可以被用来训练以及预测机载RGB图像中的单个树冠。DeepForest内部带有一个基于国家生态观测站网络(NEON : National Ecological Observatory Network)数据训练的预训练模型。在此模型基础上，用户可以注释新的数据然后训练自己的模型。DeepForest的文档是用英文编写的，如果您有兴趣为翻译文档做出贡献。欢迎与我们团队联系。

## Motivation

The original repo is written in tensorflow and can be found on pypi, conda and source (https://github.com/Weecology/DeepForest). After https://github.com/fizyr/keras-retinanet was deprecated, it became obvious that the shelf life of models that depend on tensorflow 1.0 was limited. The machine learning community is moving more towards pytorch, where many new models can be found. 

This project was just initiated. See https://github.com/weecology/DeepForest-pytorch/issues/1 to contribute to the roadmap.
