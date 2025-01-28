# Face Mask Detection Using CNN
This project implements a Face Mask Detection system using Convolutional Neural Networks (CNN) to classify images of individuals as either wearing a mask or not wearing a mask. The goal of this project is to create an effective and efficient model to detect face masks, which is crucial in the current global health scenario.

## Project Overview
The model is trained using a publicly available dataset of face mask images. The approach utilizes deep learning techniques, specifically CNNs, to analyze and classify images based on the presence of a face mask. The system is designed to be highly accurate, achieving an accuracy greater than 90% on both training and testing datasets.

## Key Features:
- **CNN Architecture**: The model utilizes layers such as Conv2D, MaxPooling2D, Flatten, Dense, and Dropout for feature extraction and classification.
- **High Accuracy**: The model achieves more than 90% accuracy on both training and testing data.
- **Overfitting Evaluation**: The model is evaluated to ensure there is no overfitting, and generalizes well to unseen data.
- **Multiple Model Exports**: The model is saved in various formats including TensorFlow SavedModel, TensorFlow Lite (TFLite), and TensorFlow.js for easy deployment.

### Technologies:
Python
TensorFlow (including TensorFlow.js, TensorFlow Lite)
OpenCV
Keras
Scikit-learn
Matplotlib
Google Colab (for dataset access)

## Project Directory
```
├───tfjs_model
|    ├───group1-shard1of1.bin
|    └───model.json
├───tflite
|    ├───model.tflite
|    └───label.txt
├───saved_model
|    ├───saved_model.pb
|    └───variables
├───notebook.ipynb
├───README.md
└───requirements.txt
```
### Directory Breakdown:
- **`tfjs_model/`**: Contains the model files for deployment using TensorFlow.js.
  - **`group1-shard1of1.bin`**: The binary weight file for the TensorFlow.js model.
  - **`model.json`**: The configuration file that contains the model architecture for TensorFlow.js.
- **`tflite/`**: Contains the TensorFlow Lite model files for deployment on mobile and embedded devices.
  - **`model.tflite`**: The model file in TensorFlow Lite format.
  - **`label.txt`**: A text file containing the labels (classes) for the model.
- **`saved_model/`**: Contains the TensorFlow SavedModel files. This is a complete TensorFlow model directory that includes both the model and its weights.
  - **`saved_model.pb`**: The model's serialized graph.
  - **`variables/`**: The directory containing the model's weights.
- **`face_mask_detection.ipynb`**: A Jupyter notebook that contains the code for training the face mask detection model and performing evaluation.
- **`README.md`**: This file, providing an overview of the project.
- **`requirements.txt`**: The list of required dependencies to run the project.

## Installation
To run this project locally, follow the steps below:

#### Step 1: Install Dependencies
Install the required libraries using the following command:
```
pip install -r requirements.txt
```

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
##### CNN Model Construction
The model uses three convolutional layers with increasing filters (32, 64, and 128) followed by max-pooling layers. After the convolutional layers, the network is completed with:
- A **Flatten** layer to reshape the output into a 1D vector.
- Two **Dense** layers, with the first having 128 units and ReLU activation, and the second being the output layer with 3 units and softmax activation.

This architecture is designed to learn hierarchical feature representations from input images and output class probabilities.

##### Model Compilation
The model is compiled with the following:
- **Optimizer**: The Adam optimizer is used to minimize the loss function.
- **Loss Function**: Categorical cross-entropy, appropriate for multi-class classification tasks.
- **Metrics**: Accuracy is tracked during training to evaluate the model's performance.

##### Early Stopping
To prevent overfitting, **Early Stopping** is applied. It monitors the validation accuracy, and if no improvement is observed for 5 consecutive epochs, training will be halted. The model will restore the best weights based on validation accuracy.

##### Model Training
- The model is trained with augmented data using **data generators** for both training and testing. This ensures that the model is exposed to varied examples, improving generalization.
- **Class Weights**: If the dataset is imbalanced, class weights are applied to give more importance to underrepresented classes.
- **Training Halting**: The training process will stop early based on validation performance, thanks to the Early Stopping callback.
This setup ensures efficient model training, minimizing overfitting while maximizing generalization performance. The architecture is well-suited for image classification tasks and is designed to perform well on a variety of input data.

#### Step 4: Model Evaluation
Evaluate the model to check for overfitting and assess its generalization performance.

#### Step 5: Save the Model
Once the model is trained and evaluated, it is saved in different formats for various deployment scenarios: SavedModel, TFLite, and TFJS.

#### Step 6: Inference
Use the trained model for inference.

#### Step 7: Deployment
Once the model is saved, you can use the appropriate format (TensorFlow, TFLite, or TensorFlow.js) for deployment based on your platform.

## Summary of Preprocessing Steps
- **Dataset Split**: The dataset is divided into training (80%) and testing (20%) sets using the `splitfolders` library.
- **Data Augmentation**: Various augmentation techniques are applied to the training set to make the model more robust. These techniques include rotation, shifting, shearing, zooming, and horizontal flipping.
- **Test Set Preprocessing**: For the test set, only rescaling is applied (no augmentation).
- **Data Generators**: Data generators are used to load and preprocess the data in batches, ensuring efficient training and evaluation.
- **Verification**: The dataset split is verified by checking the number of images in each category for both the training and test sets.

This preprocessing pipeline ensures that the model is trained on a diverse set of augmented data and is evaluated on a consistent, unaltered test set.

## Results
- **Accuracy**: The model achieved an accuracy greater than 90% on both training and testing datasets.
- **Overfitting Check**: The model has been evaluated for overfitting, and the results indicate that the model generalizes well on new data.
