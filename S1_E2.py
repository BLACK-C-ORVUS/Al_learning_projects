# LinearRegression model 
import numpy as np 
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt


F = lambda x : 4 * x + 6

Start_of_number = -10
End_of_number = 10
number_of_data = 100

X = np.linspace(Start_of_number,End_of_number,number_of_data)
Y = F(X)

Noise = np.random.randn(number_of_data) * 10
Y_noise = Y + Noise

X = X.reshape(-1,1)
Y = Y.reshape(-1,1)
Y_noise = Y_noise.reshape(-1,1)

model = LinearRegression()
model.fit( X, Y_noise)

G = lambda x : model.coef_  * x + model.intercept_
Y_Predicted = G(X)


plt.plot(X, Y, "sb")
plt.plot(X , Y_noise, " or")
plt.plot(X,Y_Predicted,"-g")
plt.title(f"Y = 4 * x + 6,Y_hat = {model.coef_} X + {model.intercept_}")
plt.legend(["Orginal Line ", "Data with Noise", "Predicted Line"])
plt.show()








