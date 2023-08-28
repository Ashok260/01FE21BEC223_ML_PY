import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from array import *
x=np.array([1,5,13,25,3]).reshape((-1,1))
y=np.array([3,6,15,12,21,18])
plt.scatter(x,y)
plt.xlabel('IV')
plt.ylabel('DV')
plt.grid()
plt.show()
from sklearn.linear_model import LinearRegression
SLR=LinearRegression()
SLR.fit(x,y)
pred=SLR.predict(x)
from sklearn.metrics import mean_squared_error
mse=mean_squared_error(y,pred)
plt.plot(x,pred,color='Red',marker='*')
plt.grid()
plt.show()
print("MSE:",mse)
print("INTERCEPT:",SLR.intercept_)
print("slope:",SLR.coef_)
