import numpy as np
from  sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt


def Creat_data ():

    F = lambda x : 2 * x + 5
    X = np.linspace( 0, 100, 101).reshape(-1, 1)
    Y = F(X).reshape(-1, 1)
    Y_noise = Y + np.random.randn(101 ,1) * 10

    X_train, X_test, Y_train, Y_test = train_test_split(
        X, 
        Y_noise, 
        test_size= 0.2, 
        random_state=69
        )
    return X_train, X_test, Y_train, Y_test 

X_train, X_test, Y_train, Y_test  = Creat_data()


def model_linear():
    model = LinearRegression()
    model.fit(X_train, Y_train)

    Y_pridicted = model.predict(X_train)
    Y_pridicted_test = model.predict(X_test)

    mse = mean_squared_error(Y_pridicted, Y_train)
    mse_test = mean_squared_error(Y_pridicted_test, Y_test)
    return mse, mse_test , Y_pridicted_test, Y_pridicted ,  model

mse, mse_test , Y_pridicted_test, Y_pridicted,  model = model_linear()


plt.subplot(1, 2, 1)
plt.plot(X_train, Y_train, "bo")
plt.plot(X_train, Y_pridicted, "g-")
plt.title(f"A : {round(model.coef_[0][0])}, B : {round(model.intercept_[0])}, MSE_TRAIN = {round(mse)} ")

plt.subplot(1, 2, 2)
plt.plot(X_test, Y_test, "bo")
plt.plot(X_test, Y_pridicted_test, "g-")
plt.title(f"A : {round(model.coef_[0][0])}, B : {round(model.intercept_[0])}, MSE_test = {round(mse_test)} ")

plt.show()