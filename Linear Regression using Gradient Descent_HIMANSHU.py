# Making the imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (5.0, 5.0)

# Preprocessing Input data
data = pd.read_csv('data.csv')
X = data.iloc[:, 0]
Y = data.iloc[:, 1]
plt.scatter(X, Y)
# plt.show()

xi = np.mean(X)
yi = np.mean(Y)

x_xi, y_yi =  X-xi , Y-yi 
# print(x_xi , y_yi)
x_xi2 , y_yi2 = x_xi * x_xi , y_yi*y_yi


#find the value of m
m = np.sum((x_xi)*(y_yi))/np.sum(x_xi2)
# print(m)

c = yi-m*xi

for i in X:
	y_list = []
	y = m*(X)+c
	y_list.append(y)

plt.scatter(X,y_list,c='r')
plt.show()

# using the Gradient Descent Method
# Building the model
m = 0
c = 0

L = 0.0001  # The learning Rate
epochs = 1000  # The number of iterations to perform gradient descent

n = float(len(X)) # Number of elements in X

# Performing Gradient Descent 
for i in range(epochs): 
    Y_pred = m*X + c  # The current predicted value of Y
    D_m = (-2/n) * sum(X * (Y - Y_pred))  # Derivative wrt m
    D_c = (-2/n) * sum(Y - Y_pred)  # Derivative wrt c
    m = m - L * D_m  # Update m
    c = c - L * D_c  # Update c
    
print (m, c)


# Making predictions
Y_pred = m*X + c

plt.scatter(X, Y) 
plt.plot([min(X), max(X)], [min(Y_pred), max(Y_pred)], color='red')  # regression line
plt.show()