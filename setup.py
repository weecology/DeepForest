from setuptools import setup, find_packages

NAME ='deepforest'
VERSION = '0.2.7'
DESCRIPTION = 'Tree crown prediction using deep learning retinanets'
URL = 'https://github.com/Weecology/DeepForest'
AUTHOR = 'Ben Weinstein'
LICENCE = 'MIT'
LONG_DESCRIPTION = """
# Deepforest

## Full documentation
[http://deepforest.readthedocs.io/en/latest/](http://deepforest.readthedocs.io/en/latest/)

## Installation

```
#Install DeepForest
pip install DeepForest

#Install fork of the retinanet repo
pip install git+git://github.com/bw4sz/keras-retinanet.git
```


Or install the latest version from Github
```
git clone https://github.com/weecology/DeepForest.git

conda env create --file=environment.yml

conda activate DeepForest
```

## Get in touch
See the [GitHub Repo](https://github.com/Weecology/deepforest) to see the
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
      python_requires='>=3',
      description=DESCRIPTION,
      long_description = LONG_DESCRIPTION,
      long_description_content_type='text/markdown',
      url=URL,
      author=AUTHOR,
      license=LICENCE,
      packages=find_packages(),
      include_package_data=True,
      install_requires=["keras > 2.3.0","tensorflow==1.15.0","pillow","pandas","opencv-python","pyyaml","slidingwindow","matplotlib","xmltodict","tqdm","numpy"],
      zip_safe=False)
