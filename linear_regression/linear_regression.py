import numpy as np 


class LinearRegression(): 

    def __init__(self, beta_0 = 0 , beta_1 = 0 ): 
        self.beta_0 = beta_0 
        self.beta_1 = beta_1 

    
    def fit(self, X, y): 
        x_mean = np.mean(X) 
        y_mean = np.mean(y) 

        self.beta_1 = self.beta_1_fit(X, x_mean, y, y_mean) 
        self.beta_0 = self.beta_0_fit(x_mean, y_mean, self.beta_1) 

        return self 
    

    def beta_0_fit(self, x_mean, y_mean, beta_1):

        beta_0 = y_mean - (beta_1 * x_mean) 

        return beta_0 
    
    def beta_1_fit(self, X, x_mean, y, y_mean): 

        beta_1 = 0 
        sum_num = 0 
        sum_den = 0 


        for x_i, y_i in zip(X, y): 
            sum_num += (x_i - x_mean)* (y_i - y_mean) 
            sum_den += (x_i - x_mean)**2
        
        beta_1 =  sum_num / sum_den 

        return beta_1 
    

    def predict(self, x): 
        value = self.beta_1 * x + self.beta_0  
        return value 
    

