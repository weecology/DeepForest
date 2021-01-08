# DeepForest-pytorch

[![Build Status](https://travis-ci.org/weecology/DeepForest-pytorch.svg?branch=master)](https://travis-ci.org/weecology/DeepForest-pytorch)

A pytorch implementation of the DeepForest model for individual tree crown detection in RGB images. DeepForest is a python package for training and predicting individual tree crowns from airborne RGB imagery. DeepForest comes with a prebuilt model trained on data from the National Ecological Observation Network. Users can extend this model by annotating and training custom models starting from the prebuilt model.

DeepForest es un paquete de python para la predicción de coronas de árboles individuales basada en modelos entrenados con imágenes remotas RVA ( RGB, por sus siglas en inglés). DeepForest viene con un modelo entrenado con datos proveídos por la Red Nacional de Observatorios Ecológicos (NEON, por sus siglas en inglés). Los usuarios pueden ampliar este modelo pre-construido por anotación de etiquetas y entrenamiento con datos locales. La documentación de DeepForest está escrita en inglés, sin embargo, agradeceríamos contribuciones con fin de hacerla accesible en otros idiomas.


## Motivation

The original repo is written in tensorflow and can be found on pypi, conda and source (https://github.com/Weecology/DeepForest). After https://github.com/fizyr/keras-retinanet was deprecated, it became obvious that the shelf life of models that depend on tensorflow 1.0 was limited. The machine learning community is moving more towards pytorch, where many new models can be found. 

This project was just initiated. See https://github.com/weecology/DeepForest-pytorch/issues/1 to contribute to the roadmap.
