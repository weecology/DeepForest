# Comparison with Other Tools

There are many open-source projects for training machine learning models.
DeepForest aims to complement these existing tools by providing a specialized, streamlined approach tailored to ecological and environmental monitoring tasks.
This document also includes an overview of the models that have been built using the DeepForest framework.
Below, we compare DeepForest with other notable tools in this space, highlighting similarities, differences, and areas of potential collaboration.

## Tools Built on DeepForest

DeepForest's core functionality has been successfully integrated into a variety of external platforms and tools. These implementations enable users to leverage DeepForest’s advanced tree crown detection capabilities across different GIS environments, all while maintaining the same model architecture and prediction performance.

### ArcGIS Tree Crown Delineation Tool

Available in the [ArcGIS Marketplace](https://www.arcgis.com/home/item.html?id=4af356858b1044908d9204f8b79ced99), this specialized plugin brings DeepForest's tree crown detection strengths into the ArcGIS Pro environment:

- Provides a seamless, native ArcGIS interface for running DeepForest predictions and analyses
- Fully integrates with Esri's comprehensive geospatial workflows, tools, and data management systems
- Requires an active, licensed ArcGIS Pro installation to operate
- Ideal for organizations and institutions already working extensively within the Esri ecosystem

### TreeEyed for QGIS

Available on [GitHub](https://github.com/afruizh/TreeEyed), this open-source QGIS plugin incorporates DeepForest together with additional models for enhanced tree crown analysis:

- Offers a flexible, open-source solution for tree crown delineation tasks
- Operates directly within the QGIS environment, providing an intuitive graphical user interface
- Combines DeepForest with supplementary segmentation, tracking, and refinement capabilities
- Suitable for users and researchers preferring robust, open-source GIS solutions without licensing restrictions

## Similar Tools

[Roboflow](https://roboflow.com) offers a comprehensive ecosystem for computer vision tasks, including tools for:

- **Supervision:** Efficient dataset annotation and augmentation.
- **Inference:** API-driven deployment of machine learning models.

The ecosystem is well-executed and widely used within DeepForest.
However, Roboflow operates as a commercial platform requiring an API key and has a range of licensing structures.
Its broad scope makes it challenging to identify robust models among thousands of projects.

**Key Differences:**

1. Roboflow is designed as an all-encompassing platform for general computer vision applications.
2. DeepForest focuses on a curated set of models tailored to ecological and environmental monitoring, offering simplicity and specificity for existing workflows.

[Torchgeo](https://github.com/microsoft/torchgeo), developed by Microsoft, is a Python library for automating remote sensing machine learning. It emphasizes:

- **Raster-based Remote Sensing:** Primarily focused on earth-facing satellite data.
- **Pretrained Models and Datasets:** Provides curated resources for remote sensing tasks.

Torchgeo caters to an audience with significant machine learning expertise and is particularly suited for satellite and aerial imagery analysis.

**Key Features:**

1. Modular design for flexibility and scalability.
2. Extensive support for raster data processing.

**Collaboration Opportunities:**

DeepForest and Torchgeo share common goals in environmental monitoring. By enhancing interoperability, both tools could enable unified workflows and reduce redundant efforts.

[AIDE](https://github.com/microsoft/aerial_wildlife_detection) is a modular web framework for annotating image datasets and training deep learning models. It integrates manual annotation and machine learning into an active learning loop:

- Humans annotate initial images.
- The system trains a model.
- The model predicts and selects additional images for annotation.

This approach accelerates tasks like wildlife surveys using aerial imagery.

**Key Features:**

- **Dual functionality**: Annotation and AI-assisted training.
- **Configurable** for various tasks, particularly ecological applications.
- **Active learning** loop for iterative model improvement.

Although AIDE has not been updated recently, it remains a powerful tool for ecological monitoring.

## Vision for Collaboration

DeepForest emphasizes the importance of collaboration in the open-source community. By connecting with tools like Roboflow, Torchgeo, and AIDE, we can:

- Standardize data formats for seamless integration.
- Share best practices for model training and deployment.
- Minimize duplication of effort and maximize community impact.

We invite users and contributors from all packages to share ideas and propose improvements to serve the community better.

**Conclusion**

The future of open-source machine learning in environmental monitoring relies on collaboration and interoperability.
Tools like DeepForest, Torchgeo, Roboflow, and AIDE complement each other, each addressing specific needs within the field.
By fostering connections between these tools, we can build a more cohesive and efficient ecosystem for solving critical environmental challenges.
