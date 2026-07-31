#_____Regression_model____
import numpy as np 
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

def f(x):
    return 2 * x + 5

def g(a, b, x):
    return a * x + b

X = np.linspace(0,100,101) 
Y = f(X)


X = X.reshape(101,1)
Y = Y.reshape(101,1)
Y_new = Y + np.random.randn(101, 1)* 100

model =LinearRegression()
model.fit(X,Y_new)
a_new = model.coef_ # A
b_new = model.intercept_ # B
Y_predicted = model.predict(X) # or
# Y_new = g(a_new, b_new, X)


print(f"A = {a_new} B = {b_new}")
plt.plot(X, Y, "b-")
plt.plot(X, Y_new, "ro")
plt.plot(X, Y_predicted, "g-")
plt.title(f"y ={a_new} * x + {b_new} ")
plt.legend(["Real line","Real Data", "Predicted Line"])
plt.show()

