# Batch Prediction in DeepForest

In this documentation, we highlight an efficient approach for batch prediction in DeepForest using the `predict_step` method and discuss a proposed enhancement to make this functionality more user-friendly.

---

## Current Challenges with Image Prediction

When working with dataloaders yielding batches of images, existing prediction methods (`predict_image`, `predict_file`, `predict_tile`) might require excessive preprocessing or manual intervention, such as:

- **Saving images to disk** to use `predict_file`.
- **Manipulating dataloaders** to ensure images are preprocessed as expected.
- Looping through each image in a batch and using `predict_image`, which is inefficient when modern GPUs can handle larger batches.

For example:
```python
for batch in test_loader:
    for image_metadata, image, image_targets in batch:
        # Preprocess image, e.g., DeepForest requires 0-255 data, channels first
        pred = m.predict_image(image)
```
This is suboptimal when GPU memory allows larger batch processing.

---

## Optimized Batch Prediction

DeepForest provides a batch prediction mechanism through the `predict_step` method. This method is part of the PyTorch Lightning framework and is intended for `trainer.predict`. While not explicitly documented, it can be leveraged directly:

### Example:
```python
for idx, batch in enumerate(test_loader):
    metadata, images, targets = batch
    # Apply necessary preprocessing to the batch
    predictions = m.predict_step(images, idx)
```
Here:
- `predict_step` processes the batch efficiently on the GPU.
- `predictions` is a list of results, formatted as dataframes, consistent with other `predict_*` methods.

---

## Limitations of Current Implementation

- **Undocumented Pathway:** The use of `predict_step` for batch predictions is not well-documented and may not be intuitive for users.
- **Reserved Method Name:** Since `predict_step` is reserved by PyTorch Lightning, it cannot be renamed. However, it can be wrapped in a user-friendly function.

---

## Proposed Solution: `predict_batch` Function

To enhance usability, we propose adding a `predict_batch` function to the API. This function would:
- Mirror the format of `predict_file`, `predict_image`, and `predict_tile`.
- Help guide users through batch prediction workflows.

### Example API for `predict_batch`
```python
def predict_batch(self, images, preprocess=True):
    """
    Predict a batch of images with the DeepForest model.

    Args:
        images: A batch of images in a numpy array or PyTorch tensor format.
        preprocess: Whether to apply preprocessing (e.g., scaling to 0-1, channels first).

    Returns:
        predictions: A list of pandas dataframes with bounding boxes, labels, and scores for each image in the batch.
    """
    # Preprocess images if required
    if preprocess:
        images = preprocess_images(images)

    # Use predict_step for efficient batch processing
    predictions = self.predict_step(images)

    return predictions
```

### Benefits:
- Streamlines batch prediction.
- Reduces the learning curve for new users.
- Ensures consistent API behavior.

---

## Next Steps

1. **Documentation:** Clearly document the current behavior of `predict_step` for advanced users.
2. **New Functionality:** Implement a `predict_batch` function to simplify batch processing workflows.
3. **Examples and Tutorials:** Add examples demonstrating batch predictions with and without the proposed `predict_batch` function.

By documenting and enhancing this functionality, DeepForest can provide a more intuitive and efficient experience for users handling large datasets.
