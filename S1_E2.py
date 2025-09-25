# Test LinearRegression model 
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt

F = lambda x : 4 * x + 2
data = 50
x = np.linspace(-10,10,data)
y = F(x)

Nois = np.random.randn(data) * 5
y_nois = y + Nois

x = x.reshape(-1,1)
y = y.reshape(-1,1)
y_nois = y_nois.reshape(-1,1)

modle = LinearRegression() 
modle.fit(x,y_nois)

g = lambda x : modle.coef_ * x + modle.intercept_
Y_predicted = g(x)
plt.plot(x,y,"sg")
plt.plot(x,y_nois,"ob")
plt.plot(x,Y_predicted,"-r")
plt.legend(["Orginal data ", "Noise data" ,"Predicted"])
plt.title(f"y = 4 * x + 2 Y_hat {modle.coef_} * x {modle.intercept_}")

plt.show()





