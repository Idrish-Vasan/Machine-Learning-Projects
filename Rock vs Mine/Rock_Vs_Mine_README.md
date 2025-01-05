# Rock vs Mine Prediction
## Project Overview
This project focuses on classifying sonar signals to determine whether an object is a rock or a mine. Using logistic regression, the model analyzes 60 features of sonar signal data to make accurate predictions.

### How It Works
- Dataset: The dataset contains 208 samples with 60 numerical features, each representing energy measurements of sonar signals.
- Goal: Build a logistic regression model to classify the signals.

## Workflow:
- Load and analyze the dataset.
- Preprocess the data by separating features and labels.
- Split the data into training and testing sets.
- Train the logistic regression model.
- Evaluate the model's accuracy on both training and testing data.
- Use the model to make predictions on new sonar signals.

## Libraries and Dependencies
```bash
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
```
### Code Workflow
## Data Loading :
The dataset is loaded into a Pandas DataFrame for analysis:
```bash
sonar_data = pd.read_csv('/content/sonar data.csv', header=None)
X = sonar_data.drop(columns=60, axis=1)
Y = sonar_data[60]
```
## Data Splitting
The dataset is split into training (90%) and testing (10%) sets:
```bash
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, stratify=Y, random_state=1)
```
## Model Training
Logistic regression is applied to the training data:
```bash
model = LogisticRegression()
model.fit(X_train, Y_train)
```
## Evaluation
The model's accuracy is evaluated on both the training and testing sets:
```bash
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
X_test_prediction = model.predict(X_test)
testing_data_accuracy = accuracy_score(X_test_prediction, Y_test)
Training Accuracy: ~83%
Testing Accuracy: ~76%
```
## Prediction System
A prediction system is implemented to classify new sonar signals:
```bash
input_data = (0.0181, 0.0146, 0.0026, 0.0141, ...)
input_data_as_numpy_array = np.asarray(input_data).reshape(1, -1)
prediction = model.predict(input_data_as_numpy_array)
```
##Concepts Covered
#### Logistic Regression
- Logistic regression is a statistical method for binary classification problems.
- It estimates probabilities using a sigmoid function and makes predictions based on a threshold value (e.g., 0.5)

####Train-Test Split
- The dataset is divided to evaluate the model's ability to generalize to unseen data.
- Accuracy Score
- This metric measures the percentage of correct predictions made by the model.

###Reflection
This project taught me:

- The importance of preprocessing data to ensure consistent results.
- How to apply logistic regression for binary classification tasks.
- Debugging skills when facing errors in data analysis and model training.