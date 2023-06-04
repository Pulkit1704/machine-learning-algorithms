import numpy as np 


class Model():
    def __init__(self, weights = None , bias = None , random_seed = None): 
        self.weights = weights 
        self.bias = bias 
        self.random_seed = random_seed 

        self.errors = []

    
    def fit(self, X, y, learning_rate = 0.01, n_iters = 10): 

        rgen = np.random.RandomState(seed = self.random_seed) 

        if self.weights == None: 
            self.weights = rgen.normal(loc = 0.0, scale = 0.01, size = X.shape[-1]) 
        
        if self.bias == None: 
            self.bias = rgen.normal(loc = 0.0, scale = 0.01, size = 1) 

        for i in range(n_iters): 
            per_iteration_error = 0 
            for x_i, y_i in zip(X, y):  

                prediction = self.predict(x_i) 

                error = y_i - prediction 

                self.weights += learning_rate * error * x_i 
                self.bias += learning_rate * error 

                # this will take care of negative errors.
                # if the error is not equal to 0, we get a boolean true
                # convering the boolean true to integer gives us 1. 
                # for every non-zero error, we get a positive number to add to the total.
                per_iteration_error += int(error != 0)
            
            self.errors.append(per_iteration_error)

        return self 
    
    def predict(self, x): 

        prediction = np.dot(self.weights, x.T) + self.bias 

        return self.activation(prediction) 

    def activation(self, y): 
        
        return np.where(y >= 0.5, 1, 0) 