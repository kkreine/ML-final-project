#!/usr/bin/env python
# coding: utf-8

#import libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#read data
data=pd.read_csv("heart.csv")
data.head()


# Data set source: https://www.kaggle.com/johnsmith88/heart-disease-dataset

#check for missing values
data.isna().sum()
#no missing values


X = data.values[:, :-1]
y = data.values[:, -1]


# In[54]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.2)


# In[55]:


print('X_train.shape=', X_train.shape)
print('X_test.shape=', X_test.shape)
print('x_train.shape[0]+x_test.shape[0]=', X_train.shape[0] + X_test.shape[0])


# ## Decision Tree (2 methods)

# In[56]:


from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

clf = DecisionTreeClassifier(criterion = 'gini')
#chose gini over entropy
clf = clf.fit(X_train, y_train)

clf.get_depth()

y_pred = clf.predict(X_train)

from sklearn.metrics import accuracy_score
print("Accuracy of Decision Tree (X_train): ", accuracy_score(y_train, y_pred)*100, "%")

y_pred = clf.predict(X_test)
print("Accuracy of Decision Tree (X_test): ", accuracy_score(y_test, y_pred)*100, "%")


# In[57]:


from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier(criterion = 'gini', max_depth=20, min_samples_leaf=8,
                            max_features='auto', random_state=2)
#chose gini over entropy

clg = clf.fit(X_train, y_train)

clf.get_depth()

y_pred = clf.predict(X_test)

from sklearn.metrics import accuracy_score
print("Accuracy of Decision Tree (test): ", accuracy_score(y_test, y_pred)*100, "%")

y_pred = clf.predict(X_train)
print("Accuracy of Decision Tree (train): ", accuracy_score(y_train, y_pred)*100, "%")


# # Random Forest

# In[58]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

forest_clf = RandomForestClassifier(max_depth=10, random_state=0) #past 10, the % stays same

forest_clf.fit(X_train, y_train)

print("Accuracy of Random Forest: ", forest_clf.score(X_test, y_test)*100, "%") 


# # Bagging

# In[59]:


from sklearn.ensemble import BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier

m = KNeighborsClassifier(n_neighbors=15)#change this
b = BaggingClassifier(m, n_estimators=5) #change up these numbers

b.fit(X_train, y_train)

print("Score = ", b.score(X_test, y_test)*100, "%") #%%


# # Decision Tree and Random Forest are the best for this data set. Bagging is not.
