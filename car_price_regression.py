import numpy as np
from sklearn.linear_model import LinearRegression

# Data: car age (years) and mileage
X = np.array([
    [1, 20000],
    [2, 35000],
    [3, 50000],
    [4, 70000],
    [5, 90000]
])

# Target: car prices
y = np.array([950, 880, 820, 760, 700])

# Create and train the Linear Regression model
model = LinearRegression()
model.fit(X, y)

# Predict price for a new car
new_car = np.array([[3, 60000]])
predicted_price = model.predict(new_car)[0]

print("Predicted Price:", predicted_price)
