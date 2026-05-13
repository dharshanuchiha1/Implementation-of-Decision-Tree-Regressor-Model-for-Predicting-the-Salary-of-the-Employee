# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the required libraries for data processing, visualization, and Decision Tree Regression.

2.Load the salary dataset and separate the input features (X) and target variable (Salary).

3.Convert categorical columns into numerical values using get_dummies() and split the dataset into training and testing sets.

4.Create and train the DecisionTreeRegressor model using the training data.

5.Display the trained Decision Tree structure using plot_tree() and show the graph using Matplotlib.
## Program:
```python
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: Dharshan G
RegisterNumber:  212225230054
*/
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, plot_tree
data = pd.read_csv("Salary (1).csv")
X = data.drop("Salary", axis=1)
y = data["Salary"]
X = pd.get_dummies(X, drop_first=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
model = DecisionTreeRegressor(random_state=42)
model.fit(X_train, y_train)
plt.figure(figsize=(25,12))
plot_tree(
    model,
    feature_names=X.columns,
    filled=True
)

plt.title("Decision Tree Regressor")
plt.show()
```

## Output:
![Decision Tree Regressor Model for Predicting the Salary of the Employee](sam.png)
<img width="553" height="458" alt="Screenshot 2026-05-13 104840" src="https://github.com/user-attachments/assets/29090be1-8f18-489f-9603-e5e6db9615e4" />
<img width="1235" height="611" alt="Screenshot 2026-05-13 104857" src="https://github.com/user-attachments/assets/b71f9d6e-2e8d-450e-a104-bbdb4501f7dd" />


## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
