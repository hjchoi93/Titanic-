#!/usr/bin/env python
# coding: utf-8

# In[1]:


import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd


# In[28]:


#Modeling techniques
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression


# In[3]:


#visualization
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import seaborn as sns


# In[29]:


from sklearn.model_selection import train_test_split


# In[31]:


#load training and test sets and append them to get total dataset
train = pd.read_csv('~/Desktop/titanic/train.csv')
test = pd.read_csv('~/Desktop/titanic/test.csv')

#entire dataset
titanic = train.append(test, ignore_index = True)
titanic.shape


# In[5]:


#conducting exploratory analysis
titanic.head()


# In[6]:


titanic.describe()


# In[7]:


#making a correlation map to see which variables are important:
def plot_correlation_map( df ):
    corr = titanic.corr()
    _ , ax = plt.subplots( figsize =( 12 , 10 ) )
    cmap = sns.diverging_palette( 220 , 10 , as_cmap = True )
    _ = sns.heatmap(
        corr, 
        cmap = cmap,
        square=True, 
        cbar_kws={ 'shrink' : .9 }, 
        ax=ax, 
        annot = True, 
        annot_kws = { 'fontsize' : 12 }
    )


# In[8]:


plot_correlation_map(titanic)


# The highest positive correlation (0.37) is seen between the # of siblings/spouses aboard with the # of parents/children aboard the Titanic. The second highest positive correlation (0.27) is between the "survived" variable and "fare." This suggests there is a reasonably strong relationship between how much a person paid for their ticket and their survival rate. Later, we will test if wealthier patrons had a higher survival rate.
# The highest negative correlation (-0.56) was between "class" and "fare." This suggests that the lower the economic status of a patron (higher the category - 1,2,3), the less chance of survival. 

# In[9]:


#plotting survival rates by variables:
def plot_categories(df , cat , target , **kwargs):
    row = kwargs.get('row' , None)
    col = kwargs.get('col' , None)
    facet = sns.FacetGrid(df , row = row , col = col)
    facet.map(sns.barplot , cat , target)
    facet.add_legend()


# In[10]:


#plotting survival rate based on class
plot_categories(titanic, cat = 'Pclass', target = 'Survived')


# The results show that the higher-class patrons are more likely to survive than lower-class patrons, which support the hypotheses yielded from the correlation matrix; that suggest lower SES people have a lesser chance of survival.

# In[11]:


#plotting survival rate by Sex
plot_categories(titanic, cat = 'Sex', target = 'Survived')


# Females have a much higher chance than males of surviving 

# In[12]:


#Data Preparation 


# We must transform some of the categorical variables into numeric variables to ensure a wide range of machine learning models can handle them. This is done so variables within the data are versatile enough for various kinds of models
# The variables Embarked, Pclass, and Sex will be changed to numerical variables. We need to create a new variable (dummy variable) for every unique value of the categorical variables. 
# The variable will hvae a value 1 if the row has that particular value, and 0 otherwise. This can also be referred to as one-hot encoding. 

# In[13]:


#Alter Sex into binary values
sex = pd.Series(np.where(titanic.Sex == 'male',1,0), name = 'Sex')


# In[14]:


#create a new variable for each unique value for port of Embarkation
embarked = pd.get_dummies(titanic.Embarked, prefix = 'Embarked')
embarked.head()


# In[15]:


pclass = pd.get_dummies(titanic.Pclass, prefix = 'Pclass')
pclass.head()


# In[16]:


#exploring missing values 
titanic.isnull().sum()


# Some models will require all variables to have values in order to use it for training the model. In this case, it makes the most sense to fill in missing values for age and fare because they are continuous variables. The simplest method is to take the average across all observations within the dataset. I am primarily focusing on demographic characteristics because I want to see what kind of people are most likely to survive. Fare is indirectly related and I am making the assumption that the higher the fare, the higher class that person is.

# In[17]:


#create datasest
filled = pd.DataFrame()
filled ['Age'] = titanic.Age.fillna(titanic.Age.mean())
filled ['Fare'] = titanic.Fare.fillna(titanic.Fare.mean())


# In[18]:


#Assemble final datasets for modeling
titanic_X = pd.concat([filled, embarked, pclass, sex], axis=1)


# In[19]:


titanic_X.head()


# In[37]:


#creating training and test sets 
train_valid_X = titanic_X[0:891]
train_valid_Y = train.Survived #dependent variable

test_X = titanic_X[891:]
train_X , valid_X , train_Y , valid_Y = train_test_split( train_valid_X , train_valid_Y , train_size = .7 )


# In[53]:


#I am going to use two different classification models
model_1 = KNeighborsClassifier(n_neighbors = 4)
model_2 = LogisticRegression()


# In[54]:


model_1.fit(train_X, train_Y)
#actual results from model
print (model_1.score( train_X , train_Y ) , model_1.score( valid_X , valid_Y ))


# In[47]:


model_2.fit(train_X, train_Y)
#actual results from model
print (model_2.score( train_X , train_Y ) , model_2.score( valid_X , valid_Y ))


# The logistic regression model is more accurate than the k-nearest neighbor model. This model would be more favored to be perform better, less likely to overfit, and generalize well to new data. 

# In[ ]:




