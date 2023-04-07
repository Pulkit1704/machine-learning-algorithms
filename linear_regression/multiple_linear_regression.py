import numpy as np 

class LinearRegression():

    def __init__(self):
        self.params = [] 

    
    def fit(self, X, y, add_intercept = True): 

        if X.shape[0] != y.shape[0]: 
            raise ValueError("no. of observations do no match the no of y values")
        
        # add a column to the data for the intercpet term. 
        if add_intercept:
            X = self._intercept(X)

        X_dot = np.dot(np.transpose(X), X) 

        X_inverse = np.linalg.inv(X_dot) 

        self.params = np.dot(X_inverse, np.dot(np.transpose(X), y))
        
        return self
    
    def _intercept(self, x):
        one = np.ones((1, x.shape[0])).T
        return np.hstack((one, x)) 
    

    def predict(self, X, add_intercept = True): 

        if add_intercept:
            X = self._intercept(X)

        y = np.dot(X, self.params) 

        return y 

