from setuptools import setup, find_packages
import setuptools
from distutils.command.build_ext import build_ext as DistUtilsBuildExt

NAME = 'deepforest'
VERSION = '1.3.3'
DESCRIPTION = 'Tree crown prediction using deep learning retinanets'
URL = 'https://github.com/Weecology/DeepForest'
AUTHOR = 'Ben Weinstein'
LICENCE = 'MIT'
LONG_DESCRIPTION = """
# Deepforest

## Full documentation
[http://deepforest.readthedocs.io/en/latest/](http://deepforest.readthedocs.io/en/latest/)

## Installation

Compiled wheels have been made for linux, osx and windows

```
#Install DeepForest
pip install deepforest
```

## Get in touch
See the [GitHub Repo](https://github.com/Weecology/DeepForest) to see the
source code or submit issues and feature requests.

## Citation

Most usage of DeepForest should cite two papers.

The first is the DeepForest paper, which describes the package:

[Weinstein, B.G., Marconi, S., Aubry‐Kientz, M., Vincent, G., Senyondo, H. and White, E.P., 2020. DeepForest: A Python package for RGB deep learning tree crown delineation. Methods in Ecology and Evolution, 11(12), pp.1743-1751. https://doi.org/10.1111/2041-210X.13472](https://doi.org/10.1111/2041-210X.13472)

The second is the paper describing the model.

For the tree detection model cite:

[Weinstein, B.G.; Marconi, S.; Bohlman, S.; Zare, A.; White, E.P., 2019. Individual Tree-Crown Detection in RGB Imagery Using Semi-Supervised Deep Learning Neural Networks. Remote Sensing 11, 1309 https://doi.org/10.3390/rs11111309](https://doi.org/10.3390/rs11111309)

For the bird detection model cite:

[Weinstein, B.G., L. Garner, V.R. Saccomanno, A. Steinkraus, A. Ortega, K. Brush, G.M. Yenni, A.E. McKellar, R. Converse, C.D. Lippitt, A. Wegmann, N.D. Holmes, A.J. Edney, T. Hart, M.J. Jessopp, R.H. Clarke, D. Marchowski, H. Senyondo, R. Dotson, E.P. White, P. Frederick, S.K.M. Ernest. 2022. A general deep learning model for bird detection in high‐resolution airborne imagery. Ecological Applications: e2694 https://doi.org/10.1002/eap.2694](https://doi.org/10.1002/eap.2694)


## Acknowledgments
Development of this software was funded by
[the Gordon and Betty Moore Foundation's Data-Driven Discovery Initiative](http://www.moore.org/programs/science/data-driven-discovery) through
[Grant GBMF4563](http://www.moore.org/grants/list/GBMF4563) to Ethan P. White.
"""

setup(name=NAME,
      version=VERSION,
      python_requires='>3.5',
      description=DESCRIPTION,
      long_description=LONG_DESCRIPTION,
      long_description_content_type='text/markdown',
      url=URL,
      author=AUTHOR,
      license=LICENCE,
      packages=find_packages(),
      include_package_data=True,
      install_requires=[
          "albumentations>=1.0.0", "aiolimiter", "aiohttp", "docformatter", "huggingface_hub", "geopandas", "matplotlib", "nbqa", "numpy",
          "opencv-python-headless>=4.5.4", "pandas", "Pillow>6.2.0", "progressbar2", "pycocotools", "pydata-sphinx-theme", "Pygments",
          "pytorch-lightning>=1.5.8", "rasterio", "recommonmark", "rtree", "scipy>1.5",
          "six", "slidingwindow", "sphinx", "supervision", "torch", "torchvision>=0.13", "tqdm",
          "xmltodict","geopandas"
      ],
      zip_safe=False)
