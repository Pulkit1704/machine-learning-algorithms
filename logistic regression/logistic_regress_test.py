import pandas as pd 
import numpy as np 
from logistic_regression import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split 


data = pd.read_csv("./datasets/Iris.csv") 

X = np.array( data.iloc[0:100, [4]]).reshape(100, 1)
standardize = StandardScaler() 
X = standardize.fit_transform(X) 

y = np.where(data.Species[0:100] == "Iris-setosa", 0,1).reshape(100, 1) 


train_x, test_x, train_y, test_y = train_test_split(X, y, test_size = 0.5) 


model = LogisticRegression() 
n_iters = 20
model = model.fit(train_x, train_y, n_iters = n_iters, alpha = 0.001) 


predictions = model.predict(test_x) 

def misclassifications(actuals, predictions): 

    misclassifications = actuals - predictions 

    return sum(misclassifications) 


print(f"no of misclassifications: {misclassifications(test_y, predictions)}")