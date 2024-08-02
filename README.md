# Crop Disease Detection - Applied Machine Learning

This repository contains the implementation of a Convolutional Neural Network (CNN) to detect plant diseases from leaf images. 

- **Version 1 (v1)**: Basic CNN model.

## Version 1 (v1)

### Description
In Version 1, a basic CNN model was developed to classify images of plant leaves into different disease categories. The model was trained on the New Plant Diseases Dataset using Keras and TensorFlow. The notebook includes the following steps:
- Data Loading and Preprocessing
- Model Architecture
- Model Training
- Model Evaluation
- Prediction on Sample Images

### Model Architecture
- Input Layer: (128, 128, 3)
- Convolutional Layer 1: 32 filters, (3, 3) kernel, ReLU activation
- MaxPooling Layer 1: (2, 2) pool size
- Convolutional Layer 2: 64 filters, (3, 3) kernel, ReLU activation
- MaxPooling Layer 2: (2, 2) pool size
- Flatten Layer
- Dense Layer 1: 128 nodes, ReLU activation
- Output Layer: Softmax activation, number of classes = 36 (number of disease categories)

### Training
- Optimizer: Adam
- Loss: Categorical Crossentropy
- Metrics: Accuracy
- Epochs: 100
- Early Stopping: Applied with patience of 5 epochs

### Results
The model achieved high accuracy on the validation set, demonstrating its capability to effectively classify plant diseases.

### Files
- `plantdiseasedetection(v.1).ipynb`: Jupyter notebook containing the code for Version 1.
- `plantdiseasedetection(v.1).html`: HTML export of the Version 1 notebook for easy viewing.
- `plant_disease_cnn_model.pkl`: Serialized model file saved using pickle.

### Data Overview
The dataset used for training and validation is the [New Plant Diseases Dataset](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset). The dataset contains images of plant leaves with and without diseases.

#### Training Data
- **Total Images**: 64,368
- **Number of Classes**: 36

#### Validation Data
- **Total Images**: 16,098
- **Number of Classes**: 36

### Getting Started

#### Prerequisites
- Python 3.x
- TensorFlow
- Keras
- NumPy
- Matplotlib
- scikit-learn

Contributing

Bhumika Pathak
Mubarak Imam
Purva Gevaria
Ricardo Patrocinio
