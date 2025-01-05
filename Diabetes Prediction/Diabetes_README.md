# Diabetes Prediction
## Project Overview

This project predicts whether a patient is diabetic based on medical data. A Support Vector Machine (SVM) classifier is trained on the PIMA Diabetes dataset to make predictions.

## How It Works
- Dataset: The dataset includes 8 features such as glucose level, BMI, age, and insulin levels, with 768 samples.

- Goal: Build an SVM classifier to predict diabetes based on patient data.

## Workflow:
- Load and preprocess the dataset.

* Standardize the features for better model performance.

- Split the data into training and testing sets.

- Train the SVM model with a linear kernel.

- Evaluate the model’s accuracy on both datasets.

- Build a predictive system for new patient data.

## Libraries and Dependencies
```bash
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
```
Code Workflow

### Data Loading
The dataset is loaded into a Pandas DataFrame:
```bash
diabetes_dataset = pd.read_csv('/content/diabetes.csv')
X = diabetes_dataset.drop(columns='Outcome', axis=1)
Y = diabetes_dataset['Outcome']
```
### Data Standardization
Standardization ensures all features contribute equally to the model:
```bash
scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)
```
### Data Splitting
The dataset is split into training (80%) and testing (20%) sets:
```bash
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
```
### Model Training
An SVM classifier with a linear kernel is trained on the data:
```bash
classifier = svm.SVC(kernel='linear')
classifier.fit(X_train, Y_train)
```
### Evaluation
The model’s accuracy is evaluated on both the training and testing sets:
```bash
X_train_prediction = classifier.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
X_test_prediction = classifier.predict(X_test)
testing_data_accuracy = accuracy_score(X_test_prediction, Y_test)
```
Training Accuracy: ~78%Testing Accuracy: ~77%

### Prediction System

A prediction system is implemented to classify new patient data:
```bash
input_data = (8, 95.53, 29.16, 81.15, ...)
input_data_as_numpy_array = np.asarray(input_data).reshape(1, -1)
std_data = scaler.transform(input_data_as_numpy_array)
prediction = classifier.predict(std_data)
```
## Concepts Covered
### Support Vector Machine (SVM)
- SVM is a supervised learning algorithm that finds the optimal hyperplane to separate data points into distinct classes.
- A linear kernel is used for simplicity and efficiency in this project.

### Standardization
Standardization rescales features to have a mean of 0 and a standard deviation of 1, ensuring equal contribution from all features.

### Accuracy Score
Measures the percentage of correct predictions made by the model on the dataset.

## Reflection
This project helped me:
- Understand the importance of feature scaling in improving model accuracy.
- Apply SVM for a practical classification problem.
- Gain experience handling class imbalance and evaluating models effectively.