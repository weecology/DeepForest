# Prebuilt models

At the moment, DeepForest has two prebuilt models: Bird Detection and Tree Crown Detection.

## Tree Crown Detection

The model was initially described in [Remote Sensing](https://www.mdpi.com/2072-4292/11/11/1309) on a single site. The prebuilt model uses a semi-supervised approach in which millions of moderate quality annotations are generated using a LiDAR unsupervised tree detection algorithm, followed by hand-annotations of RGB imagery from select sites. Comparisons among geographic sites were added to [Ecological Informatics](https://www.sciencedirect.com/science/article/pii/S157495412030011X). The model was further improved, and the Python package was released in [Methods in Ecology and Evolution](https://besjournals.onlinelibrary.wiley.com/doi/full/10.1111/2041-210X.13472).

![image](../www/MEE_Figure4.png)

### Citation
> Weinstein, B.G.; Marconi, S.; Bohlman, S.; Zare, A.; White, E. Individual Tree-Crown Detection in RGB Imagery Using Semi-Supervised Deep Learning Neural Networks. Remote Sens. 2019, 11, 1309

## Bird Detection

The model was initially described in [Ecological Applications](https://esajournals.onlinelibrary.wiley.com/doi/abs/10.1002/eap.2694). From the abstract

>
 Using over 250,000 annotations from 13 projects from around the world, we develop a general bird detection model that achieves over 65% recall and 50% precision on novel aerial data without any local training despite differences in species, habitat, and imaging methodology. Fine-tuning this model with only 1000 local annotations increases these values to an average of 84% recall and 69% precision by building on the general features learned from other data sources. 
 >


 ### Citation
> Weinstein, B.G., Garner, L., Saccomanno, V.R., Steinkraus, A., Ortega, A., Brush, K., Yenni, G., McKellar, A.E., Converse, R., Lippitt, C.D., Wegmann, A., Holmes, N.D., Edney, A.J., Hart, T., Jessopp, M.J., Clarke, R.H., Marchowski, D., Senyondo, H., Dotson, R., White, E.P., Frederick, P. and Ernest, S.K.M. (2022), A general deep learning model for bird detection in high resolution airborne imagery. Ecological Applications. Accepted Author Manuscript e2694. https://doi-org.lp.hscl.ufl.edu/10.1002/eap.2694

![image](../www/example_predictions_small.png)

```
#Load deepforest model and set bird label
m = main.deepforest(label_dict={"Bird":0})
m.use_bird_release()
```

![](../www/bird_panel.jpg)

We have created a [GPU colab tutorial](https://colab.research.google.com/drive/1e9_pZM0n_v3MkZpSjVRjm55-LuCE2IYE?usp=sharing
) to demonstrate the workflow for using the bird model.

For more information, or specific questions about the bird detection, please create issues on the [BirdDetector repo](https://github.com/weecology/BirdDetector)

## Want more pretrained models?

Please consider contributing your data to open source repositories, such as zenodo or lila.science. The more data we gather, the more we can combine the annotation and data collection efforts of hundreds of researchers to built models available to everyone. We welcome suggestions on what models and data are most urgently [needed](https://github.com/weecology/DeepForest/discussions). 