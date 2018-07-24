#polynomial regression

#imorting libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#importing dataset
dataset=pd.read_csv('Position_Salaries.csv')
x=dataset.iloc[:,1:2].values
y=dataset.iloc[:,2].values

#splitting to linear and test set
""""from sklearn.cross_validation import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
"""

#feature scaling
"""from sklearn.preprocessing import StandardScaler
sc_x=StandardScaler()
x_train=sc_x.fit_transform(x_train)
x_test=sc_x.transform(x_test)
sc_y=StandardScaler()
y_train=sc_y.fit_transform(y_train)"""

#3fitting Linear regression to data set
from sklearn.linear_model import LinearRegression
lin_reg=LinearRegression()
lin_reg.fit(x,y)

#fitting polynomial regression to data set
from sklearn.preprocessing import PolynomialFeatures 
poly_reg= PolynomialFeatures(degree=4)
x_poly=poly_reg.fit_transform(x)
lin_reg2=LinearRegression()
lin_reg2.fit(x_poly,y)

#visualizing linear regression result
plt.scatter(x,y,color='red')
plt.plot(x,lin_reg.predict(x),color='blue')
plt.title('truth or buff')
plt.xlabel('level')
plt.ylabel('salary')
plt.show()

#visualizing polynomial regression in dataset
plt.scatter(x,y,color='red')
plt.plot(x,lin_reg2.predict(poly_reg.fit_transform(x)),color='blue')
plt.title('truth or buff polynomial regression')
plt.xlabel('level')
plt.ylabel('salary')
plt.show()

# predicting new result with linear regression
lin_reg.predict(6.5)

#predicting new result with polynomial regression
lin_reg2.predict(poly_reg.fit_transform(6.5))
