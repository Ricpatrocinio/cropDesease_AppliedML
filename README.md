# Plant Disease Detection Project

## Description

This project aims to develop a convolutional neural network (CNN) to detect various plant diseases from images. The project consists of two versions with different levels of complexity and data augmentation techniques.

## Version 1

### Overview
- **Title**: Plant Disease Detection (v.1)
- **Training Data**: Basic data augmentation with `rescale=1./255`
- **Model Architecture**:
  - Input layer
  - Two convolutional layers with 32 and 64 filters respectively, each followed by a max pooling layer
  - Flatten layer
  - Dense layer with 128 units and ReLU activation
  - Output dense layer with softmax activation for classification
- **Training**:
  - Optimizer: Adam
  - Loss: Categorical Crossentropy
  - Metrics: Accuracy
  - Early stopping with patience of 5 epochs
- **Results**: Model achieved a high accuracy on validation data.
- **Saved Model**: `plant_disease_cnn_model.pkl`

### Key Files
- `plantdiseasedetection(v.1).ipynb`
- `plant_disease_cnn_model.pkl`
- `README.md`

## Version 2

### Overview
- **Title**: Enhanced Plant Disease Detection (v.2)
- **Training Data**: Advanced data augmentation including:
  - Rescaling
  - Rotation
  - Width shift
  - Height shift
  - Shear
  - Zoom
  - Horizontal flip
- **Model Architecture**:
  - Input layer
  - Three convolutional layers with 32, 64, and 128 filters respectively, each followed by a max pooling layer
  - Dropout layers for regularization after each max pooling layer
  - Flatten layer
  - Dense layer with 256 units and ReLU activation
  - Dropout layer
  - Output dense layer with softmax activation for classification
- **Training**:
  - Optimizer: Adam
  - Loss: Categorical Crossentropy
  - Metrics: Accuracy
  - Early stopping with patience of 5 epochs
- **Results**: Model achieved a high accuracy on validation data.
- **Saved Model**: `enhanced_plant_disease_cnn_model.pkl`

### Key Files
- `plantdiseasedetection(v.2).ipynb`
- `enhanced_plant_disease_cnn_model.pkl`
- `README.md`

## Contributing
Special thanks to the contributors:
- [Purva Gevaria](https://github.com/purvagevaria/)
- [Bhumika Pathak](https://github.com/BhumikaPathak2)
- [Mubarak Imam](https://github.com/Mubarak-Imam)
- [Ricardo Patrocinio](https://github.com/Ricpatrocinio)

## Dataset
The dataset used in this project can be found on [Kaggle](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset).

## How to Run
1. Clone this repository: `git clone https://github.com/Ricpatrocinio/cropDesease_AppliedML.git`
2. Navigate to the project directory.
3. Open the desired version's notebook (`.ipynb` file) and run the cells sequentially to train and evaluate the model.
4. Use the saved model for predictions on new images as shown in the notebook.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
