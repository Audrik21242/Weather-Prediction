# Weather Prediction from Images
This project builds a Convolutional Neural Network (CNN) model to classify weather conditions based on images, including categories such as rain, hail, snow, and more. The model is intended to aid in the automatic identification of weather patterns, which can be helpful in applications like forecasting, emergency management, and climate analysis.

# Table of Contents
1. Project Overview
2. Dataset
3. Model Architecture
4. Data Preprocessing
5. Training Process
6. Technologies Used
7. Deployment
8. Results

# Project Overview
The goal of this project is to create a deep learning model capable of identifying weather conditions from images. By using Convolutional and Pooling layers, the model learns to distinguish visual features associated with various weather types. A web application interface allows users to upload weather images for classification, making the tool easily accessible.

# Dataset
The dataset consists of images labeled with specific weather conditions, such as rain, snow, fog, and hail. Due to the dataset's large size, images are processed in batches and split into training, validation, and test sets.

# Model Architecture
The model is built using a Convolutional Neural Network (CNN) architecture, which is particularly effective for image classification tasks:
 * Layers: Multiple convolutional and pooling layers capture spatial features from the images.
 * Augmentation: Images are resized, rotated, and transformed to increase variety and model robustness under different conditions.
 * Batch Processing: Training was done in batches, ensuring efficient computation on large datasets.

# Data Preprocessing
Data preprocessing steps include:
 * Image Resizing and Augmentation: Each image is resized and augmented to simulate different orientations and lighting conditions, helping the model generalize better.
 * Caching and Batch Loading: Caching was used to reduce latency during training, and data was loaded in batches to optimize memory usage.
 * Data Splitting: The dataset was divided into training, validation, and test sets for reliable model evaluation.

# Training Process
The model was trained using TensorFlow (v2.10), utilizing tensorflow-directml for efficient training on compatible hardware. The training process involved:
 * Batch Training: Due to the dataset size, training was conducted in batches to make optimal use of computational resources.
 * Evaluation Metrics: The model was evaluated based on accuracy, loss, and prediction confidence.

# Technologies Used
 * Programming Languages: Python, HTML, CSS
 * Data Manipulation: NumPy, Pandas
 * Model Building: TensorFlow (v2.10) with DirectML
 * Visualization: Matplotlib
 * Other Utilities: os for model saving
 * Deployment: Flask

# Deployment
A Flask-based web application serves as the user interface:
 * Frontend: Users can upload an image of a weather scene through a simple HTML form.
 * Backend: The image is processed by the model, which returns the predicted weather condition along with the confidence level.
 * Output: The app displays the predicted weather category and its confidence percentage to the user.

# Results
The CNN model shows promising accuracy in identifying various weather conditions based on image inputs.
