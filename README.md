# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. PREPARE YOUR DATA:

Collect and clean data on employee salaries and features

Split data into training and testing sets

2. Define your model
Use a Decision Tree Regressor to recursively partition data based on input features

Determine maximum depth of tree and other hyperparameters

3. Train your model
Fit model to training data

Calculate mean salary value for each subset

4. Evaluate your model
Use model to make predictions on testing data

Calculate metrics such as MAE and MSE to evaluate performance

5. Tune hyperparameters
Experiment with different hyperparameters to improve performance

6. Deploy your model
Use model to make predictions on new data in real-world application.

## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: BHARATHWAJ R
RegisterNumber: 212222240019
*/

import pandas as pd
data = pd.read_csv("dataset/Salary.csv")
data.head()

data.info()

data.isnull().sum()

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data["Position"] = le.fit_transform(data["Position"])
data.head()

x = data[["Position", "Level"]]
x.head()

y = data["Salary"]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 2)

from sklearn.tree import DecisionTreeRegressor
dt = DecisionTreeRegressor()
dt.fit(x_train, y_train)
y_pred = dt.predict(x_test)

from sklearn import metrics
mse = metrics.mean_squared_error(y_test, y_pred)
mse

r2 = metrics.r2_score(y_test, y_pred)
r2

dt.predict([[5, 6]])
```

## Output:
Initial dataset:

![image](https://github.com/BHARATHWAJRAMESH/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/119394248/1f31c366-1fa9-4ad0-aa7e-725b6a7b17c4)



Data Info:

![image](https://github.com/BHARATHWAJRAMESH/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/119394248/f92b726f-1101-4277-a9f2-fd1734fa46a1)



Optimization of null values:

![image](https://github.com/BHARATHWAJRAMESH/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/119394248/4adbf48d-f004-4522-81d7-17c14043fefc)



Converting string literals to numericl values using label encoder:

![image](https://github.com/BHARATHWAJRAMESH/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/119394248/86d4f1ea-9329-4910-8f19-ce16119144e8)



Assigning x and y values:

![image](https://github.com/BHARATHWAJRAMESH/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/119394248/f1d7c1d3-40a9-4af4-8ace-df25af44c536)



Mean Squared Error:

![image](https://github.com/BHARATHWAJRAMESH/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/119394248/b17a6d62-0f31-46f9-87b4-86c17330fa5e)



R2 (variance):

![image](https://github.com/BHARATHWAJRAMESH/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/119394248/d89ea6e0-6dea-423f-a448-8312714beacf)



Prediction:

![image](https://github.com/BHARATHWAJRAMESH/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/119394248/c4b08179-0892-4278-aa9f-0d5f957315e4)



## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
