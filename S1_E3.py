# Linear Regression model for the California Housing dataset
from  sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

dataset = fetch_california_housing()

X = dataset["data"][:,4].reshape(-1,1)
Y = dataset["target"].reshape(-1,1)

number_of_train_data = int(len(X) * 0.8)
number_of_test_data = len(X) - number_of_train_data

train_data = X[0: number_of_train_data, : ]
train_labels = Y[0: number_of_train_data, : ]

test_data = X[number_of_train_data: , :]
test_labels = Y[number_of_train_data: , :]

model = LinearRegression()
model.fit(train_data, train_labels)

train_predictions = model.predict(train_data)
train_error = mean_squared_error(train_labels, train_predictions)

test_predictions = model.predict(test_data)
test_error = mean_squared_error(test_labels, test_predictions)
print(f"Train Error : {train_error}, Test Error : {test_error}")

plt.plot(train_data, train_labels, "ob")
plt.plot(train_data, train_predictions, "sr")
plt.show()

