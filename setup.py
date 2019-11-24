from setuptools import setup, find_packages

NAME ='deepforest'
VERSION = 0.0.4
DESCRIPTION = 'Tree Crown Prediction using Deep Learning Retinanets'
URL = 'https://github.com/Weecology/DeepForest'
AUTHOR = 'Ben Weinstein'
LICENCE = 'MIT'
LONG_DESCRIPTION = """
# Deepforest

## Full documentation  
[http://deepforest.readthedocs.io/en/latest/](http://deepforest.readthedocs.io/en/latest/)
## Installation

conda create --file=environment.yml

```
Or install the latest version from Github  
```
pip install git+git://github.com/Weecology/deepforest
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
      description=DESCRIPTION,
      long_description = LONG_DESCRIPTION,
      long_description_content_type='text/markdown',
      url=URL,
      author=AUTHOR,
      license=LICENCE,
      packages=find_packages(),
      include_package_data=True,
      zip_safe=False)