# Image-Dimensionality-Reduction-and-Face-Recognition-using-PCA

# Overview

This assignment delves into Principal Component Analysis (PCA) and its application in reducing dimensionality for facial image data. The primary objective is to utilize PCA to reduce the dimensionality of facial images and build a face recognition system based on this reduced feature space.

# Learning Objectives

1. **Theory of PCA:** Understand the underlying concepts and theory behind Principal Component Analysis.
2. **PCA in Image Data:** Apply PCA for reducing dimensionality in image data.
3. **Face Recognition System:** Implement a face recognition system using the reduced feature space.
4. **Experience with ML and Image Datasets:** Gain hands-on experience in working with machine learning techniques and image datasets.

# Instructions

Task 1: Data Preparation

1. **Dataset Creation:** Prepare a dataset of at least 5 family members, each with a minimum of 10 preprocessed grayscale images, standardized to a fixed size and center-aligned using libraries like OpenCV or PIL.
2. **Data Split:** Divide the dataset into training and testing sets, reserving 20% of images per person for testing.

Task 2: Principal Component Analysis (PCA)

1. **Implement PCA:** Create a PCA algorithm from scratch using NumPy and libraries to compute eigenvalues and eigenvectors.
2. **Selecting Principal Components:** Experiment with different numbers of principal components, leveraging explained variance ratios for informed dimensionality reduction.

Task 3: Face Recognition

1. **Feature Extraction:** Project training and testing data onto the reduced feature space using selected eigenvectors.
2. **Classifier:** Utilize the k-Nearest Neighbors algorithm for face recognition with reduced feature vectors.
3. **Evaluation:** Assess the face recognition system's performance on testing data, considering metrics like accuracy, precision, recall, and F1-score.


# Report
Principle Component Analysis (PCA) is a technique used for dimensionality reduction whilst  retaining the important features of the dataset. It transforms the data into a new coordinate  system by identifying its most correlated features. KNN is a classifier used for prediction based on the reduced feature space. The dataset used in this assignment is of five celebrities and public figures: Kate Middleton,  Imran Khan, Tom Cruise, Rowan Atkinson, Barack Obama. The dataset contains 10 pictures of each of them, so in total I have data of 50 pictures each labeled like IK1,IK2,….IK10 for  Imran Khan’s pictures, KM1,KM2,..,KM10 for Kate Middleton’s pictures and so and so  forth. 

For the preprocessing of images used OpenCV then I converted the images to grayscale and then standardized them to fixed size. The data was then split into training and testing data.20% of images per person were set aside for testing while 80% images per person were used  for training purposes. PCA was implemented using Scikit-learn’s PCA module. The ‘fit’ function was applied to  training dataset to learn the transformation from data and then used the learned  transformation to transform both the training and testing data into the reduced feature space  using ‘transform’. Feature Extraction was done by projecting the training and testing data  onto the reduced feature space obtained from PCA, utilizing the transformed data. Classifier  Selection employed the k-Nearest Neighbors (KNN) classification algorithm to recognize  faces based on the reduced feature vectors. Trained the KNN classifier using the reduced  feature space of the training dataset. Utilized the trained model to predict labels for the test  set and evaluated the model's performance using metrics like accuracy, precision, recall, and F1-score. The overall accuracy achieved was 60%.

Classification report:

![image](https://github.com/AmnaAkbar1/Image-Dimensionality-Reduction-and-Face-Recognition-using-PCA/assets/145672191/574fbfca-cf1e-4c9c-acfd-983a6f4dc08a)

Confusion Matrix:

![image](https://github.com/AmnaAkbar1/Image-Dimensionality-Reduction-and-Face-Recognition-using-PCA/assets/145672191/42f00fb3-eb20-4b18-8a99-673e9ac03ede)

The KNN classifier performed well for classes like 'BO', achieving perfect precision and  recall. However, for classes like 'IK', 'KM', 'RA', 'TC', the performance was relatively poor,  showing low precision, recall, and F1-scores. Further improvement is required, especially for  classes with low performance, possibly by acquiring more diverse data or refining the  classifier. While the overall accuracy of the model is moderate, it exhibits variability in  performance across different classes. Further optimization and data augmentation might enhance the classifier's capability to recognize individuals more accurately




