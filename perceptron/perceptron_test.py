import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from perceptron import Model 
from sklearn.model_selection import train_test_split 
from sklearn.metrics import mean_absolute_error 


data = pd.read_csv("./datasets/Iris.csv") 

X = data.iloc[:100, [1, 3]].to_numpy() 

y = np.where(data.Species[:100] == "Iris-setosa", 0, 1)

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.1) 

ppr = Model(random_seed = 1111)

ppr_trained = ppr.fit(x_train, y_train) 

plt.plot(range(len(ppr_trained.errors)), ppr_trained.errors, marker = 'o', label = "per iteration error") 
plt.xlabel("no. of iterations") 
plt.ylabel("training error") 
plt.title("perceptron model convergence") 
plt.legend() 
plt.grid(True) 
plt.show() 

test_preds = ppr_trained.predict(x_test) 

test_error = mean_absolute_error(y_test, test_preds) 

print(f"mean absolute error: {test_error}")

print(f"true test data labels: {y_test}")

print(f"predicted test data labels: {test_preds}")
