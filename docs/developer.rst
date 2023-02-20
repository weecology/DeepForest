=================
Developer's guide
=================

Deepends on Python 3.5+

Getting started
===============

1. Quickstart by forking the main repository https://github.com/weecology/DeepForest
2. Clone your copy of the repository

    - Using ssh ``git clone git@github.com:[your user name]/DeepForest.git``
    - Using https ``git clone https://github.com/[your user name]/DeepForest.git``

3. Link or point your cloned copy to the main repository. (I always name it upstream)

    - ``git remote add upstream https://github.com/weecology/DeepForest.git``

5. Check/confirm your settings using ``git remote -v``

::

    origin  git@github.com:[your user name]/DeepForest.git (fetch)
    origin  git@github.com:[your user name]/DeepForest.git (push)
    upstream  https://github.com/weecology/DeepForest.git (fetch)
    upstream  https://github.com/weecology/DeepForest.git (push)


6. Install the package from the main directory.

Deepforest can be install using either pip or conda.

**Pip**

Installing with Pip uses `dev_requirements`_.txt.

.. code-block:: bash

  $ pip install -r dev_requirements.txt
  $ pip install . -U

**conda**

Installing with Conda uses `environment yaml`_.

Conda-based installs can be slow. We recommend using [mamba](https://mamba.readthedocs.io/en/latest/user_guide/mamba.html) to speed them up.

.. code-block:: bash

  $ conda create -n deepforest python=3
  $ conda activate deepforest
  $ pip install . -U


7. Check if the package was installed, please test using the `sample code`_.

Testing
=======

Running tests locally
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

  $ pip install . --upgrade or python setup.py install
  $ pytest -v

Testing Conda Deepforest build
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We use the conda_recipe/meta.yaml to make sure that conda build can build the package

.. code-block:: bash

  $ cd conda_recipe
  $ conda build conda_recipe/meta.yaml -c conda-forge -c defaults


Conda staged recipe update
^^^^^^^^^^^^^^^^^^^^^^^^^^

Update Conda recipe after every release.

Clone the `Weecology staged recipes`_.
Checkout deepforest branch update the `deepforeset/meta.yaml` with the new version and the sha256 values.
Sha256 values are obtained from the source on
`PYPI download files`_. using the deepforest-{version-number}.tar.gz 

.. code-block::

  {% set version = "fill new" %}
  {% set sha256 = "fill new" %}


Documentation
=============

We are using `Sphinx`_. and `Read the Docs`_. for the documentation.

**Update Documentation**

The documetation is automatically updated for changes with in functions.
However, the documentation should be updated after addition of new functions or modules.

Change to the docs directory and use sphinx-apidoc to update the doc's ``source``.
Exclude the tests and setup.py documentation

Run

.. code-block:: bash

  sphinx-apidoc -f  -o ./source ../ ../tests/* ../setup.py

The ``source`` is the destination folder for the source rst files. ``../`` is the path to where
the deepforest source code is located relative to the doc directory.

**Test Documentation locally**

.. code-block:: bash

  cd  docs  # go the docs directory and install the current changes pip install ../ -U
  make clean # Run
  make html # Run

  Note:
  Do not commit the build directory after making html.

.. _sample code: https://github.com/weecology/DeepForest#usage
.. _dev_requirements: https://raw.githubusercontent.com/weecology/DeepForest/main/dev_requirements.txt
.. _environment yaml: https://raw.githubusercontent.com/weecology/DeepForest/main/environment.yml
.. _Python download site: http://www.python.org/download/
.. _PYPI download files: https://pypi.org/project/deepforest/#files
.. _Weecology staged recipes: https://github.com/weecology/staged-recipes
.. _Conda staged recipes: https://github.com/conda-forge/staged-recipes
.. _Sphinx: http://www.sphinx-doc.org/en/stable/
.. _Read The Docs: https://readthedocs.org//
