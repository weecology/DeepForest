from setuptools import setup, find_packages
import setuptools
from distutils.command.build_ext import build_ext as DistUtilsBuildExt

NAME = 'deepforest-pytorch'
VERSION = '0.1.35'
DESCRIPTION = 'Tree crown prediction using deep learning retinanets'
URL = 'https://github.com/Weecology/DeepForest-pytorch'
AUTHOR = 'Ben Weinstein'
LICENCE = 'MIT'
LONG_DESCRIPTION = """
# Deepforest

## Full documentation
[http://deepforest-pytorch.readthedocs.io/en/latest/](http://deepforest-pytorch.readthedocs.io/en/latest/)

## Installation

Compiled wheels have been made for linux, osx and windows

```
#Install DeepForest-pytorch
pip install deepforest-pytorch
```

## Get in touch
See the [GitHub Repo](https://github.com/Weecology/DeepForest-pytorch) to see the
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
          "torch", "torchvision", "matplotlib", "Pillow>6.2.0", "pandas", "progressbar2", "six", "scipy>1.5", "slidingwindow","geopandas",'rasterio','rtree',"pytorch_lightning",
          "tqdm", "xmltodict", "numpy","imagecodecs","albumentations"
      ],
      zip_safe=False)
