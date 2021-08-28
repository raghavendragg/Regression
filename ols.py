#!/usr/bin/env python
# coding: utf-8

# ## Linear Regression with OLS
# 
# The notebook contains incomplete code for the problem along with the necessary information in the form of comments that helps you to complete the project.  
# 
# ## Step 1 - Importing the required libraries 
# 
# We have completed this step for you. Please go through it to have a clear idea about the libraries that are used in the project.

# In[1]:


import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
# Remember what this line did?
get_ipython().run_line_magic('matplotlib', 'inline')
import math 

# the library we will use to create the model 
from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split
from sklearn import metrics 


# ## Step 2 - Reading the dataset and splitting it into testing and training data

# In[2]:


# 2.1 Read the dataset from the csv file using pandas and write into a pandas dataframe object named 'dataset'
dataset = pd.read_csv('father_son_heights.csv')
dataset.head


# Now we need to process the data so that the sklearn function for generating the model can be invoked. This involves converting the data into numpy arrays and reshaping. 
# Please note that reshaping is required only when the array is single dimensional and we will have to reshape it in this case as there is only one independent variable.
# 

# In[3]:


x = dataset['Father'].values.reshape(-1,1)
y = dataset['Son'].values.reshape(-1,1)
x_size = x.shape
print(x_size)
y_size = y.shape
print(y_size)


# In[4]:


# 2.2 Split the data into test and train data using the function train_test_split imported above. 
# Recommended test size is 0.3. 
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=50)
x_train_size = x_train.shape
print(x_train_size)
y_train_size = y_train.shape
print(y_train_size)


# ## Step 3 - Visualization of data 
# 
# Now that we have our data split into train and test, we will visualize the data through a scatterplot. 
# Use matplotlib to plot the graph. 

# In[5]:


# 2.3 Make a scatterplot of the data using matplotlib
# Plot
plt.scatter(x, y, s=9.42, c='r', alpha=1)
plt.title('Fathers and Sons heights scatter plot')
plt.xlabel('Fathers height')
plt.ylabel('Sons height')
#plt.show()
plt.savefig('scatterplot.png')


# ## Step 4 - Generating the model 
# 
# Now that we have visualized the data, we proceed to generate a model for the same. 
# Here we need to make use of the linear_model library we imported above and then 'fit' the training data to the model object. 

# In[6]:


# 2.4 generate a model object using the library imported above 
linearRegressor = LinearRegression()
linearRegressor.fit(x_train, y_train)


# Now we obtain the intercept and coefficient values for our generated model using the model object. 
# The intercept and coefficient of the simple regression model are obtained as properties of the model object. 

# In[7]:


# 2.6 print('Predicted coeffient value is: ', )
coeff = len(linearRegressor.coef_)
print('Predicted coeffient value is: ', coeff)


# 2.7 print('Predicted intercept value is: ', )
intercept = linearRegressor.intercept_
print('Predicted intercept value is: ', intercept)


# In[8]:


print("Predicted coeffient value is: ", coeff, file=open("output.txt", "a"))
print("Predicted intercept value is: ", intercept, file=open("output.txt", "a"))


# ## Step 5 - Visualizing the line obtained
# 
# Having built the model, let us plot it.
# 
# 1. The scatterplot will be plotted the in the same way as done in step 3. 
# 2. Plotting the line requires to input a list of points (x,y) to the matplotlib function.
# 3. For this, make use of linspace() function to generate 50 points between the minimum and maximum values of x in the given data. The function 'linspace()' is part of numpy. 
# 4. We get the predicted values (y) of these 50 points (x) using the predict() which is a function of the model obtained above. You may have to reshape the list of 50 points here too, as required by the library.  
# 5. We feed this data to the matplotlib function and obtain our line. 
# 

# In[9]:


# code for the line here 
# minimum_x_test = min(x_test)
# maximum_x_test = max(x_test)
x_new = np.linspace( min(x_test), max(x_test), 50)
x_newReshape = x_new.reshape(-1,1)
yPrediction = linearRegressor.predict(x_newReshape)
plt.scatter(x, y, color = 'red')
plt.plot(x_new, yPrediction, color = 'blue')
plt.title('Fathers and Sons heights line and scatter plot')
plt.xlabel('Fathers height')
plt.ylabel('Sons height')
plt.savefig('line_and_scatter.png')


# In[ ]:





# In[ ]:





# In[ ]:




