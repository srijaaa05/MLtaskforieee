# CIFAR-10 Image Classification with Convolutional Neural Network (CNN)

This project demonstrates an image classification pipeline using a Convolutional Neural Network (CNN) to classify images in the CIFAR-10 dataset. CIFAR-10 contains 60,000 color images in 10 different classes (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck), making it an excellent dataset for learning image classification with deep learning.

## Project Overview

1. *Dataset*:
   - We used the CIFAR-10 dataset, divided into training and testing sets.
   - The dataset contains images of 32x32 pixels across 10 classes.
   - Images are loaded using TensorFlow's ImageDataGenerator, which includes rescaling.

2. *Model Architecture*:
   - The CNN model consists of three convolutional layers followed by max-pooling layers for feature extraction.
   - Two dense layers are added at the end, with a softmax layer for classification into 10 categories.
   - The model uses the ReLU activation function for non-linearity and categorical cross-entropy loss function.

3. *Training and Evaluation*:
   - The model is trained for 10 epochs with training and validation accuracy/loss tracked to visualize learning.
   - The Adam optimizer is used to update the network weights, ensuring efficient learning.
   - After training, the model is evaluated on the test dataset to determine the final test accuracy.

4. *Results Visualization*:
   - The accuracy and loss metrics are plotted for both training and validation sets to understand the model’s learning behavior.
   - A confusion matrix is generated to show the classification performance for each category.

## Requirements

- Python
- TensorFlow
- NumPy
- Matplotlib
- Seaborn

## How to Run

1. Ensure the dataset is structured as specified in train_dir and test_dir.
2. Run the script to train the model and visualize results.

## Future Improvements

- Explore deeper CNN architectures for improved accuracy.
- Use data augmentation to enhance generalization.
- Experiment with dropout layers to reduce overfitting.
