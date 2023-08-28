import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
data=pd.DataFrame
data=pd.read_csv(r"C:\Users\L.SUNITHA\Downloads\data.csv")
x=np.array(data.loc[0:9,("x")]).reshape((-1,1))
y=np.array(data.loc[0:9,("y")])
print(x)
print(y)
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
