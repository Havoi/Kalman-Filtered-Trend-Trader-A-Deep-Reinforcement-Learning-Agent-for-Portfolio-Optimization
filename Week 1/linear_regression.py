import numpy as np 
import matplotlib.pyplot as plt

X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
Y = np.array([12, 24, 29, 45, 52, 63, 75, 82, 94, 98])


def mean_squared_error(y_true , y_predicted):
    n = len(y_true)
    cost =  np.sum((y_true - y_predicted)**2)/n
    return cost 

def gradient_descent(X , Y , m , c , L):
    n = len(X)
    y_pred = m*X + c
    
    dm = -(2/n) * np.sum(X * (Y - y_pred))
    dc = -(2/n) * np.sum(Y - y_pred )
    
    m = m - (L* dm)
    c= c - (L * dc)
    
    return m ,c



def OLS(X , Y):
    n =  len(Y)
    Y_bar = np.mean(Y)
    X_bar = np.mean(X)
    num = np.sum((X - X_bar)*(Y-Y_bar))
    den = np.sum((X - X_bar)**2)
    m = num/den
    c = Y_bar - (m*X_bar)
    return m , c

m = 0
c = 0 

L = 0.01
cost_history = []
epochs = int(input("no of training steps : "))
for i in range(epochs):
    m , c = gradient_descent(X , Y , m , c, L)
    current_cost = mean_squared_error(Y , m * X + c)
    cost_history.append(current_cost)
    # Create the prediction line using our new w and b
    # Print progress every 100 steps
    if i % 100 == 0:
        print(f"Iteration {i}: Cost = {current_cost:.2f}")
        

print(f" from Gradient descent m and c are : {m} and {c}")
m_ols , c_ols = OLS(X , Y)
print(f" from OLS m and c are : {m_ols} and {c_ols}")
regression_line = m * X + c

# # Plot the original data points
# plt.scatter(X, Y, color='blue', label='Actual Data')

# # Plot our new best-fit line
# plt.plot(X, regression_line, color='red', label='Prediction Line')

# plt.title("Linear Regression from Scratch")
# plt.xlabel("Hours Studied")
# plt.ylabel("Exam Score")
# plt.legend()
# plt.show()
# plt.plot(range(epochs), cost_history, color='green')
# plt.title("Cost Function over Iterations")
# plt.xlabel("Number of Iterations")
# plt.ylabel("Cost (Mean Squared Error)")
# plt.show()