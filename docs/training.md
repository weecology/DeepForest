# Training new models - An Overview

Our work has shown that starting training from the prebuilt model increases performance, regardless of the geographic location of your data. In the majority of cases, it will be useful for the model to have learned general tree representations that can be refined using hand annotated data.

## Design evaluation data

In our experience, defining a clear evaluation dataset and setting a threshold for desired performance is critical before training. It is common to just dive into training new data with only a vague sense of the desired outcome. This is always a mistake. We highly suggest users spend the time to answer 2 questions.

* What kind of data am I trying to predict?

Capturing the variability and the broad range of tree taxonomy and presentation will make development go more smoothly.

* What kind of accuracy do I need to answer my question?

It is natural to want the best model possible, but one can waste a tremendous amount of time trying to eek out another 5% of recall without understanding whether that increase in performance will improve our ecological or natural resource questions. Prioritize evaluation data that matches your desired outcomes.

## Gather annotations

DeepForest uses xml files produced by the commonly used annotation program RectLabel. Please note that Rectlabel is an inexpensive program available only for Mac.

Rectlabel screenshot

xml format explanation


### Guidelines for model Training
