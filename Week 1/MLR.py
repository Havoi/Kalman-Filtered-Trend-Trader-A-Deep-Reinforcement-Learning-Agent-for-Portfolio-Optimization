import numpy as np
import matplotlib.pyplot as plt

X  = np.array([ [1 , 2 , 4 , 5 , 7] , [16 , 19 , 21 , 19 , 13] , [94 ,100 , 131 , 129 , 107]])
Y = np.array([67 , 78 , 89 , 92, 76])
X = X.T
X_raw = X
# plt.scatter(X[: , 0] , Y )
# plt.show()
# Normalization is also needed since the data is varying in magnitude to get all the data in a similar range of -1 and 1
# We subtract the mean and divide by standard deviation for each column
mean = np.mean(X , axis = 0 )
std = np.std(X , axis = 0)
X = (X-mean) / std
# Gradient descent for multiple weights

w1 = 0 
w2 = 0 
w3 = 0 
b = 0 
W = [w1 , w2 , w3]
L = 0.01
# line  :  y = w1x1  + w2x2 + w3x3+b 
cost_history = []
def gradient_descent(W, b ,L,X,Y ):
    n = len(Y)
    y_pred = X@W + b 
    dw = (-2/n)*((Y-y_pred).T @ X)
    W = W - L*dw
    db = (-2/n)*(np.sum(Y-y_pred))
    b = b - L*db
    cost = np.mean((Y - y_pred)**2)
    cost_history.append(cost)
    return W , b

iter = input("Number of Iterations : ")
iter = int(iter)

for i in range(iter):
    W ,b = gradient_descent(W , b , L , X , Y)
    
y_pred = X@W + b


# plot of cost over iterations
plt.plot(range(iter), cost_history)
plt.xlabel('Iterations')
plt.ylabel('Cost (MSE)')
plt.title('Gradient Descent Progress')
plt.show()


# plot of each feature against predicted Y
plt.figure(figsize=(10, 6))

# 2. Plot the Actual Y values as blue dots
plt.scatter(X_raw[ : , 2], Y, color='blue', label='Actual Y')

# 3. Plot the Predicted Y values as a red line (or dots)
plt.scatter(X_raw[: , 2], y_pred, color='red' , label = 'predicted Y')

# Formatting
plt.title('Actual vs Predicted Values')
plt.xlabel('Feature Value')
plt.ylabel('Value')
plt.legend()
plt.grid(True)
plt.show()