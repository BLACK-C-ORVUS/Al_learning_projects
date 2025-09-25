import matplotlib.pyplot as plt
import numpy as np

F = lambda x : 4 * x + 2
 
x = np.linspace(-10 ,10 ,10)

y = F(x)

plt.plot(x , y, "-b")
plt.show()


