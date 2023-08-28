import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from array import *
a=[]
b=[]
n=int(input("Enter the number of elements"))
for i in range(n):
    c=np.random.randint(0,15)
    d=np.random.randint(0,15)
    a.append(c)
    b.append(d)
print(a,b)

x=np.array(a).reshape((-1,1))
y=np.array(b)
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
