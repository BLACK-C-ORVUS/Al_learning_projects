import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def f(X):
    return 2 * X + 5


def Create_data():
    X = np.linspace(0,100,101)
    Y = f(X)
    X = X.reshape(-1,1)
    Y = Y.reshape(-1,1)
    Y_nois = Y +  np.random.randn(101, 1) * 10
    
    x_train, x_test, y_train , y_test = train_test_split(
        X, 
        Y_nois,
        test_size=0.2,
        random_state=69)
    
    return x_train, x_test, y_train , y_test



x_train, x_test, y_train , y_test = Create_data()


model = LinearRegression()
model.fit(x_train , y_train) #Learning

Y_predic = model.predict(x_train) #Data prediction with the new model 
mse = mean_squared_error( y_train, Y_predic)

y_test_pred = model.predict(x_test)
mse_test = mean_squared_error(y_test, y_test_pred)


plt.subplot(1,2,1)

plt.plot(x_train, y_train, "bs")
plt.plot(x_train, Y_predic, "g-")
plt.title(f"y ={round(model.coef_[0][0])} * x + {round(model.intercept_[0]) } Error: {round(mse)}")
plt.legend(["Real Data", "Predicted Line"])


plt.subplot(1,2,2)
plt.plot(x_test, y_test, "bs")
plt.plot(x_test, y_test_pred, "g-")
plt.title(f"y ={round(model.coef_[0][0])} * x + {round(model.intercept_[0]) } Error: {round(mse_test)}")
plt.legend(["Real Data", "Predicted Line"])


plt.show()