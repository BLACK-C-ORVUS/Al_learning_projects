#numpy & matplotlib
import numpy as np 
import matplotlib.pyplot as plt


F =  lambda x : 4 * x + 6

Start_number = -10
End_number = 10
number_of_data = 100
X = np.linspace(Start_number,End_number,number_of_data)

Y = F(X)

plt.plot(X , Y , "ob")
plt.show()