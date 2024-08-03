# -Classification-of-Retinal-OCT-Image-using-Convolutional-Transformer-Approach


# Retinal OCT Image Classification using Convolutional-Transformer

## Introduction
This project focuses on classifying retinal OCT (Optical Coherence Tomography) images using a combination of Convolutional Neural Networks (CNN) and Transformer architectures. The goal is to accurately categorize retinal OCT images into different classes, improving diagnostic accuracy for eye conditions.

## Dataset
The dataset used in this project consists of retinal OCT images, categorized into 4 classes i,e. Diabetic Macular Edema (DME), Choroidal Neovascularization (CNV), Druesen, Normal. The dataset is preprocessed and split into training, validation, and test sets.
-- No. of images in training set= 25600 (6400 in each classes)
-- No. of images in testing set= 6400  (1600 in each classes)
-- No. of images in Validation set= 6400  (1600 in each classes)

Dataset source:- https://www.kaggle.com/datasets/paultimothymooney/kermany2018

## Model Architecture
The model used in this project combines CNNs and Transformers to leverage both local feature extraction and global context understanding. The CNN component is used for initial feature extraction, followed by a Transformer for capturing long-range dependencies.

## Training
The model is trained using the following key steps:
1. Data augmentation is applied to the training set to increase the robustness of the model.
2. The model is trained using a 'Adam' optimizer with learning rate= 0.0003 and loss function CrossEntropyLoss.
3. The training process includes validation to monitor performance and prevent overfitting.

## Evaluation
The model's performance is evaluated using various metrics:
- Accuracy: Overall accuracy on the test set.
- ROC Curve: Receiver Operating Characteristic curve to evaluate the model's discriminative ability.

## Results
The model achieved the following performance metrics on the test set:
-Test loss: 0.3523
-Test accuracy: 87.69 %

## Dependencies & Setup
Libraries used in this projects are:
- Python 
- PyTorch 
- NumPy
- Matplotlib
- Scikit-learn
- Other relevant libraries





## Acknowledgments
- This project is inspired by recent advancements in medical imaging and deep learning.
- Special thanks to the contributors of the dataset and the open-source community.


