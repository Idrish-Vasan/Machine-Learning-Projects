# -*- coding: utf-8 -*-
"""Rock  vs Mine.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1rprDxJXyU1S23448_Wjv4sgKaHziLwkq

Importing the Dependecies
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

"""Data Collection and Data Processing

"""

#loading the dataset into dataframe
sonar_data=pd.read_csv('/content/sonar data.csv',header=None)

sonar_data.head()

sonar_data.shape

sonar_data.describe()   #statiscal data

sonar_data[60].value_counts()



"""M --> Represents Mine

R --> Represents Rock
"""

sonar_data.groupby(60).mean()

X=sonar_data.drop(columns=60,axis=1)
Y=sonar_data[60]

print(X,Y)

"""Training and Testing Data"""

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.1,stratify=Y,random_state=1)

print(X.shape,X_test.shape,X_train.shape)

"""Model Training --->
Logistic Regression
"""

model=LogisticRegression()

print(X_train,Y_train)

model.fit(X_train,Y_train)

"""Model Evaluation"""

#accuracy
X_train_prediction=model.predict(X_train)
training_data_accuracy=accuracy_score(X_train_prediction,Y_train)

print('Accuracy on training data :',training_data_accuracy)

X_test_prediction=model.predict(X_test)
testing_data_accuracy=accuracy_score(X_test_prediction,Y_test)
print('Accuracy on testing data :',testing_data_accuracy)

"""Making a Prediction System"""

input_data=(0.0181,0.0146,0.0026,0.0141,0.0421,0.0473,0.0361,0.0741,0.1398,0.1045,0.0904,0.0671,0.0997,0.1056,0.0346,0.1231,0.1626,0.3652,0.3262,0.2995,0.2109,0.2104,0.2085,0.2282,0.0747,0.1969,0.4086,0.6385,0.7970,0.7508,0.5517,0.2214,0.4672,0.4479,0.2297,0.3235,0.4480,0.5581,0.6520,0.5354,0.2478,0.2268,0.1788,0.0898,0.0536,0.0374,0.0990,0.0956,0.0317,0.0142,0.0076,0.0223,0.0255,0.0145,0.0233,0.0041,0.0018,0.0048,0.0089,0.0085)
#changing the input_data to a numpy_array
input_data_as_numpy_array=np.asarray(input_data)

#reshaping the np array as we are predicting for one instance
input_data_reshaped=input_data_as_numpy_array.reshape(1,-1)
prediction=model.predict(input_data_reshaped)
#printing the prediction as Mine or Rock
print(prediction)
if(prediction=='R'):
  print("It is a Rock")
else:
  print("It is a Mine")

