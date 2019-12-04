# FAQ

Commonly encountered issues

1. Alpha channel

```
OSError: cannot write mode RGBA as JPEG
```

If you are manually cropping an image, be careful not to save the alpha channel. For example, on OSX, the preview tool will save a 4 channel image (RGBA) instead of a three channel image (RGB) by default. When saving a crop, toggle alpha channel off.

2. Prediction versus training models


3.
