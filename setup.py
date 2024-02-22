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
If you use this software in your research please cite it as:

Geographic Generalization in Airborne RGB Deep Learning Tree Detection
Ben. G. Weinstein, Sergio Marconi, Stephanie A. Bohlman, Alina Zare, Ethan P. White
bioRxiv 790071; doi: https://doi.org/10.1101/790071


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
          "albumentations>=1.0.0",
          "geopandas",
          "imagecodecs",
          "matplotlib",
          "numpy",
          "opencv-python>=4.5.4",
          "pandas",
          "Pillow>6.2.0",
          "progressbar2",
          "pycocotools",
          "pytorch-lightning>=1.5.8",
          "rasterio",
          "recommonmark",
          "rtree",
          "scipy>1.5",
          "six",
          "slidingwindow",
          "sphinx",
          "torch",
          "torchvision>=0.13",
          "tqdm",
          "xmltodict"
      ],
      zip_safe=False)
