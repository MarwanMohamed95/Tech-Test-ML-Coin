**Coin Detection with U-Net**

## Task

Welcome to the "Coin Detection with U-Net" project! In this project, we wil make implementation of coin detection using semantic segmentation.
we used two different model:
1. The first one is by building a U-Net model from scratch.
2. The second one is by using transfer learning by using VGG16 pre-trained model for the encoder part, implemented and trained the decoder part.

### Below is a general overview about the structure of every notebook (method)
**Project Overview:**

1. **Data Preparation:** We'll start by loading and preparing our dataset, which includes images of coins and their corresponding masks. Applying Data preprocessing for the images and masks.

2. **U-Net Architecture:** Here we implemented the U-Net model.

3. **Model Training:** Training the model. We'll train our U-Net on the coin dataset, monitor its performance.

4. **Prediction and Post-processing:** Once our model is trained, we'll use it to make predictions on new coin images. We'll also apply post-processing techniques, such as morphological operations, to refine our coin segmentations.

5. **Visualization:** Visualizing the results is essential for understanding how well our model is performing. We'll create visualizations that show both the input images and the model's coin predictions.

6. **Coin Detection in Action:** We'll apply our trained model to real-world scenarios, demonstrating its ability to detect and segment coins accurately.

**Why U-Net?**

U-Net is a popular choice for image segmentation tasks because of its architecture, which combines contracting and expansive pathways, making it highly effective at capturing fine-grained details in images.
