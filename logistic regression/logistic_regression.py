import numpy as np 

# logistic regression model for binary classification. 

class LogisticRegression(): 

    def __init__(self, weights = None, bias = None , random_seed = None):

        self.weights = weights
        self.bias = bias 
        self.random_seed = random_seed
        self.errors = [] 


    def fit(self, X, y, n_iters = 10, alpha = 0.01):

        # apply the gradient descent method to update weights. 

        #initialize parameters 
        if self.weights == None: 
            rgen = np.random.RandomState(self.random_seed) 
            self.weights  = rgen.normal(0, 0.1, X.shape[1]) 
        
        if self.bias == None: 
            self.bias = 0
        
        # iteration 
        for _ in range(n_iters): 

            cost = self.cost(y , X)
            self.errors.append(cost)

            predictions = self.predict(X) 

            mis_classifications = y - predictions

            self.weights -= alpha * sum(np.dot(X.T, mis_classifications))
            self.bias -= alpha * sum(mis_classifications) 

        return self 
    
    def predict(self, X):

        y = self.sigmoid(X)
        
        return np.where(y>=0.5, 0, 1) 
    
    def sigmoid(self, x):

        z = self.linear_equation(x)

        return  1/(1 + np.exp(-z)) 
    
    def linear_equation(self, x):

        y = (self.weights * x) + self.bias 
        return y 

    def cost(self, y, X): 
        
        rig_result = self.linear_equation(X) 

        cost1 = np.dot(y.T, np.log(np.exp(rig_result))) 
        cost2 = np.dot((1 - y).T, np.log(1 + np.exp(rig_result)))

        return sum((cost1 + cost2)) 