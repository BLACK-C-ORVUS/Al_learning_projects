import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

def f(X):
    return 2 * X + 5

X = np.linspace(0,100,101)
Y = f(X)
X = X.reshape(101,1)
Y = Y.reshape(101,1)
Y_nois = Y +  np.random.randn(101, 1) * 10

model = LinearRegression()
model.fit(X , Y_nois)

Y_predic = model.predict(X)
mse = mean_squared_error(Y_predic, Y_nois)

plt.plot(X, Y_nois, "b-")
plt.plot(X, Y_nois, "ro")
plt.plot(X, Y_predic, "g-")
plt.title(f"y ={model.coef_} * x + {model.intercept_} Error: {mse} ")
plt.legend(["Real line","Real Data", "Predicted Line"])
plt.show()