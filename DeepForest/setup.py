import setuptools

setuptools.setup(
    name="DeepForest",
    version="0.1.0",
    url="https://github.com/weecology/macrosystems",

    author="Ben Weinstein",
    author_email="ben.weinstein@weecology.org",

    description="Segmentation of tree crowns using airborne lidar, orthophotos and hyperspectral data for NEON sites",
    long_description=open('README.rst').read(),

    packages=setuptools.find_packages(),

    install_requires=[],

    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Programming Language :: Python',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
    ],
)
