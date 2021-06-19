#!/usr/bin/env python
# coding: utf-8

# # The Sparks Foundation
# 
# ### Data Science and Business Analytics Internship Task 1
# 
# ### Task by: Abinash R
# 
# ## Prediction using Supervised ML
# 
# #### Step 1: Understand the Problem
# 
# Predict the percentage of a student based on the no. of study hours.
# 
# #### Step 2: Find the source of data and load it:
# 
# Dataset is available at: https://raw.githubusercontent.com/AdiPersonalWorks/Random/master/student_scores%20-%20student_scores.csv

# In[6]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[9]:


df = pd.read_csv("https://raw.githubusercontent.com/AdiPersonalWorks/Random/master/student_scores%20-%20student_scores.csv")


# In[10]:


df.head()


# #### Step 3: Exploring the Data

# In[13]:


df.info()


# In[14]:


df.describe()


# #### Step 4: Visualizing the Data 

# In[15]:


plt.scatter(df.Hours, df.Scores)
plt.title("Hours Vs Scores")
plt.xlabel(" Hours >")
plt.ylabel(" Scores >")
plt.grid()
plt.show()


# #### Step 5: Model Selection and Training 

# In[16]:


# Taking input column in x
x = df['Hours']

# Taking output column in y
y = df['Scores']


# In[17]:


x.head()


# In[18]:


y.head()


# In[19]:


# Splitting the data into train and test for model building
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state = 0)


# In[20]:


len(x_train), len(x_test)


# In[22]:


from sklearn.linear_model import LinearRegression
model = LinearRegression()


# In[23]:


x_train = x_train.values.reshape(-1, 1)
y_train = y_train.values.reshape(-1, 1)
x_test = x_test.values.reshape(-1, 1)
y_test = y_test.values.reshape(-1, 1)


# In[24]:


model.fit(x_train, y_train)


# In[26]:


# Plotting the mx + c line over test data

m = model.coef_[0]
c = model.intercept_


x_line = np.arange(0, 10, 0.1)

y_line = m * x_line + c

plt.plot(x_line, y_line, "r")

plt.scatter(x_test, y_test)
plt.show()


# #### Step 6: Testing the Model 

# In[27]:


model.score(x_test, y_test)


# In[29]:


y_pred = model.predict(x_test)


# #### What will be predicted score if a student studies for 9.25 hrs/ day? 

# In[31]:


user_input = np.array(9.25).reshape(1, -1)
answer = model.predict(user_input)
print(answer)


# ### Result - Predicted score if a student studies for 9.25 hrs/ day is "93.89272889341655". 
