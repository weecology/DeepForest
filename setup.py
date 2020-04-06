from setuptools import setup, find_packages
import setuptools
from setuptools.extension import Extension
from distutils.command.build_ext import build_ext as DistUtilsBuildExt

NAME ='deepforest'
VERSION = '0.2.12'
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
pip install DeepForest
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

#DeepForest wraps of fork of keras-retinanet, see https://github.com/fizyr/keras-retinanet/blob/master/setup.py

class BuildExtension(setuptools.Command):
    description     = DistUtilsBuildExt.description
    user_options    = DistUtilsBuildExt.user_options
    boolean_options = DistUtilsBuildExt.boolean_options
    help_options    = DistUtilsBuildExt.help_options

    def __init__(self, *args, **kwargs):
        from setuptools.command.build_ext import build_ext as SetupToolsBuildExt

        # Bypass __setatrr__ to avoid infinite recursion.
        self.__dict__['_command'] = SetupToolsBuildExt(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._command, name)

    def __setattr__(self, name, value):
        setattr(self._command, name, value)

    def initialize_options(self, *args, **kwargs):
        return self._command.initialize_options(*args, **kwargs)

    def finalize_options(self, *args, **kwargs):
        ret = self._command.finalize_options(*args, **kwargs)
        import numpy
        self.include_dirs.append(numpy.get_include())
        return ret

    def run(self, *args, **kwargs):
        return self._command.run(*args, **kwargs)


extensions = [
    Extension(
        'keras_retinanet.utils.compute_overlap',
        ['keras_retinanet/utils/compute_overlap.pyx']
    ),
]

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
      cmdclass         = {'build_ext': BuildExtension},      
      install_requires=["keras > 2.3.0","keras-resnet==0.1.0","six","scipy","tensorflow==1.14.0","Pillow","pandas","opencv-python","pyyaml","slidingwindow","matplotlib","xmltodict","tqdm","progressbar2"],
      ext_modules    = extensions,
      setup_requires = ["cython>=0.28", "numpy>=1.14.0"],    
      zip_safe=False)
