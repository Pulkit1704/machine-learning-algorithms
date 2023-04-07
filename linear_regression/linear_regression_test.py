import numpy as np 
import matplotlib.pyplot as plt 
from linear_regression import LinearRegression 


def line_point(x, slope = 3, intercept = 2): 

    # straight line point generator. 

    y_value = intercept + slope *x

    return y_value


def randomize(values): 
    new_values = [] 
    rgen = np.random.RandomState(1) 

    for i in values: 
        new_values.append(i + rgen.normal(0, 10))
    
    return new_values 
    


if __name__ == '__main__': 

   x = [i for i in range(10)] 

   y_true = [line_point(i) for i in x] 

   y_random = randomize(y_true) 

   plt.scatter(x, y_random, label = 'random point') 
   plt.plot(x, y_true, color ='orange', label = 'true function') 
   plt.xlabel("x values") 
   plt.ylabel("y value") 

   model = LinearRegression() 

   model = model.fit(x, y_random) 

   print(model.beta_0, model.beta_1) 

   predictions = [model.predict(i) for i in x]

   plt.plot(x, predictions, color = 'red', label = "model fit") 
   plt.legend() 
   plt.show() 
