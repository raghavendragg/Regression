# Import required library
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
# Remember what this line did?
# %matplotlib inline  
import math 

# the library we will use to create the model 
from sklearn.linear_model import LinearRegression

# the library used to spilt data into train and test data
from sklearn.model_selection import train_test_split
from sklearn import metrics

# Import data
dataset = pd.read_csv('father_son_heights.csv')

# Independent variable
x = dataset['Father'].values.reshape(-1,1)
# Dependent variable
y = dataset['Son'].values.reshape(-1,1)
X = np.array(x)
print(X)

# Spilt data
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3)

# Plot
plt.scatter(x,y,color='blue')
plt.show()

reg = LinearRegression().fit(x_train,y_train)

print('Predicted coeffient value is: ', reg.coef_)
print('Predicted intercept value is: ', reg.intercept_)

x_lin = np.linspace(x.min(),x.max()).reshape(-1,1)
plt.scatter(x,y,c='b')
plt.plot(x_lin, reg.predict(x_lin), c='r')
plt.xlabel('Father\'s Height (inch)')
plt.ylabel('Son\'s Height (inch)')
plt.show()