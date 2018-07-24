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

#fitting  regression model to data set
#create regressor


#predicting new result with regression
y_pred=regressor.predict(6.5))

#visualizing regression in dataset
plt.scatter(x,y,color='red')
plt.plot(x,regressor.predict(x)),color='blue')
plt.title('truth or buff  regression')
plt.xlabel('level')
plt.ylabel('salary')
plt.show()

#visualizing random forrest regression in dataset (for high resolution)
x_grid=np.arange(min(x),max(x),0.1)
x_grid=x_grid.reshape(len(x_grid),1)
plt.scatter(x,y,color='red')
plt.plot(x_grid,regressor.predict(x_grid),color='blue')
plt.title('truth or buff  random forrest regression')
plt.xlabel('level')
plt.ylabel('salary')
plt.show()
