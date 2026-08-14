
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pylab as plt

def create_data():
    F = lambda X : 2 * X + 5 

    X = np.linspace(0, 100, 101).reshape(-1, 1)
    Y = F(X).reshape(-1, 1)

    Y_nois = Y + np.random.randn(101, 1) * 10

    X_train, X_test, Y_train ,Y_test  = train_test_split(
        X, 
        Y_nois,
        test_size= 0.2,
        random_state= 69
    )
    return X_train, X_test, Y_train ,Y_test


X_train, X_test, Y_train ,Y_test = create_data()

model= LinearRegression()
model.fit(X_train, Y_train)

Y_predict = model.predict(X_train)
Y_predict_test = model.predict(X_test)

mse = mean_squared_error(Y_train, Y_predict)
mse_test = mean_squared_error(Y_test, Y_predict_test)

plt.subplot(1, 2, 1)
plt.plot(X_train, Y_train, "bp")
plt.plot(X_train, Y_predict, "g-")
plt.title(f"A = {round(model.coef_[0][0], 2)}, B = {round(model.intercept_[0], 2)},  MSE_TRAIN = {round(mse, 2)}")
plt.legend(["Train Line ", "Predict line"])

plt.subplot(1, 2, 2)
plt.plot(X_test, Y_test, "bp")
plt.plot(X_test, Y_predict_test, "g-")
plt.title(f"A = {round(model.coef_[0][0], 2)}, B = {round(model.intercept_[0], 2)},  MSE_TRAIN = {round(mse, 2)}")
plt.legend(["Test line ", "Tets Prrdite lin"])

plt.show()
