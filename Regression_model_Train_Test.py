# Linear Regression Model: Training and Testing Results
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt




def Create_date():
    f = lambda X: 2 * X + 5
    
    X = np.linspace(0,100,101).reshape(-1,1)
    Y = f(X).reshape(-1,1)
    
    Y_noise = Y + np.random.randn(101, 1) * 10
    
    X_train, X_test, Y_train, Y_test = train_test_split(
        X,
        Y_noise,
        test_size=0.2,
        random_state=69)
    
    return X_train, X_test, Y_train, Y_test


X_train, X_test, Y_train, Y_test = Create_date()


model = LinearRegression()
model.fit(X_train, Y_train)

Y_predict =model.predict(X_train)
Y_predict_test = model.predict(X_test)

mse = mean_squared_error(Y_train, Y_predict)
mse_test = mean_squared_error(Y_test, Y_predict_test)

plt.subplot(1, 2, 1)
plt.plot(X_train,Y_train, "bo")
plt.plot(X_train, Y_predict, "g-")
plt.title(f"A = {round(model.coef_[0][0])}, B = {round(model.intercept_[0])}, MSE_TRAIN = {round(mse)}")

plt.subplot(1, 2, 2)
plt.plot(X_test,Y_test, "bs")
plt.plot(X_test, Y_predict_test, "g-")
plt.title(f"A = {round(model.coef_[0][0])}, B = {round(model.intercept_[0])}, MSE_test = {round(mse_test)}")
plt.show()

