# Face Mask Detection Using CNN
This project implements a Face Mask Detection system using Convolutional Neural Networks (CNN) to classify images of individuals as either wearing a mask or not wearing a mask. The goal of this project is to create an effective and efficient model to detect face masks, which is crucial in the current global health scenario.

## Project Overview
The model is trained using a publicly available dataset of face mask images. The approach utilizes deep learning techniques, specifically CNNs, to analyze and classify images based on the presence of a face mask. The system is designed to be highly accurate, achieving an accuracy greater than 90% on both training and testing datasets.

## Key Features:
**CNN Architecture**: The model utilizes layers such as Conv2D, MaxPooling2D, Flatten, Dense, and Dropout for feature extraction and classification.
**High Accuracy**: The model achieves more than 90% accuracy on both training and testing data.
**Overfitting Evaluation**: The model is evaluated to ensure there is no overfitting, and generalizes well to unseen data.
**Multiple Model Exports**: The model is saved in various formats including TensorFlow SavedModel, TensorFlow Lite (TFLite), and TensorFlow.js for easy deployment.

### Technologies:
Python
TensorFlow (including TensorFlow.js, TensorFlow Lite)
OpenCV
Keras
Scikit-learn
Matplotlib
Google Colab (for dataset access)

## Installation
To run this project locally, follow the steps below:

#### Step 1: Install Dependencies
Install the required libraries using the following command:
``` pip install -r requirements.txt ```
Here’s a quick guide on the key libraries used:
`tensorflow, tensorflowjs, opencv-python, split-folders, scikit-learn, matplotlib`

#### Step 2: Download Dataset
To access the dataset, you need to use your Kaggle account. Upload the Kaggle API key file (kaggle.json) to this environment.

```
from google.colab import files
files.upload()
```
Once uploaded, the dataset is fetched and extracted using the Kaggle API:
```
!pip install kaggle
!mkdir -p ~/.kaggle
!mv kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json
!kaggle datasets download -d shiekhburhan/face-mask-dataset
The dataset will be extracted into the FMD_DATASET folder.
```
#### Step 3: Train the Model
Once the dataset is prepared, you can begin training the model.
The model will be trained using the dataset of images, with a split for training and testing. The accuracy should exceed 90% after training.

#### Step 4: Model Evaluation
Evaluate the model to check for overfitting and assess its generalization performance.

#### Step 5: Save the Model
Once the model is trained and evaluated, it is saved in different formats for various deployment scenarios: SavedModel, TFLite, and TFJS.

#### Step 6: Inference
Use the trained model for inference.

#### Step 7: Deployment
Once the model is saved, you can use the appropriate format (TensorFlow, TFLite, or TensorFlow.js) for deployment based on your platform.

## Results
Accuracy: The model achieved an accuracy greater than 90% on both training and testing datasets.
Overfitting Check: The model has been evaluated for overfitting, and the results indicate that the model generalizes well on new data.
