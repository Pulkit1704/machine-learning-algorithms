import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split 
from multiple_linear_regression import LinearRegression 


data = pd.read_csv("/home/pulkit/Programming/datasets/Advertising.csv", index_col = "Unnamed: 0")


X = np.array(data[['TV', 'radio']]) 
y = np.array(data.sales) 

train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.3) 


model = LinearRegression() 
model = model.fit(train_x, train_y) 
print(f"manual model parameters are: {model.params}")


predictions = model.predict(test_x) 


from sklearn.metrics import mean_absolute_error 

error = mean_absolute_error(test_y, predictions) 


print(f"manual model error: {error}")

ax = plt.axes(projection = '3d') 

ax.scatter(test_x[:, 0], test_x[:, -1], test_y) 
ax.scatter(test_x[:, 0], test_x[:, -1], predictions, color= 'red') 


plt.show() 

