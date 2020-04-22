#!/usr/bin/env python
# coding: utf-8

# # ALL THE LIBRARIES ARE TO BE IMPORTED 

# In[53]:


from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
import numpy as np
import pandas as pd
import math
import json
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.externals import joblib
import scipy.sparse
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
import warnings; warnings.simplefilter('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')


# # DATASET IS UPLOADED

# In[54]:


# Import the dataset and give the column names
columns=['userId', 'productId', 'ratings','timestamp']
electronics_df=pd.read_csv('ratings_Electronics.csv',names=columns)


# In[55]:


electronics_df.head()


# In[56]:


electronics_df.drop('timestamp',axis=1,inplace=True)


# In[57]:


electronics_df.info()


# In[58]:


#Check the number of rows and columns
rows,columns=electronics_df.shape
print('Number of rows: ',rows)
print('Number of columns: ',columns)


# In[59]:


#Check the datatypes
electronics_df.dtypes


# In[60]:


#Taking subset of the dataset
electronics_df1=electronics_df.iloc[:50000,0:]


# Since the data is very big. Consider electronics_df1 named dataframe with first 50000 rows and all columns from 0 of dataset.

# In[61]:


electronics_df1.info()


# In[62]:


#Summary statistics of rating variable
electronics_df1['ratings'].describe().transpose()


# In[63]:


#Find the minimum and maximum ratings
print('Minimum rating is: %d' %(electronics_df1.ratings.min()))
print('Maximum rating is: %d' %(electronics_df1.ratings.max()))


# The ratings in the given dataset ranges from 1 to 5

# # We Check and analyse for the missing value

# In[64]:


#Check for missing values
print('Number of missing values across columns: \n',electronics_df.isnull().sum())


# In[65]:


# Check the distribution of the rating
with sns.axes_style('white'):
    g = sns.factorplot("ratings", data=electronics_df1, aspect=2.0,kind='count')
    g.set_ylabels("Total number of ratings")


# The above graph shows the total now of reviews or ratings given by users and the ratings given

# In[66]:


# Number of unique user id  in the data
print('Number of unique users in Raw data = ', electronics_df1['userId'].nunique())
# Number of unique product id  in the data
print('Number of unique product in Raw data = ', electronics_df1['productId'].nunique())


# the above shows user id and product id

# In[67]:


#Check the top 10 users based on ratings
most_rated=electronics_df1.groupby('userId').size().sort_values(ascending=False)[:10]
print('Top 10 users based on ratings: \n',most_rated)


# In[68]:


counts=electronics_df1.userId.value_counts()
electronics_df1_final=electronics_df1[electronics_df1.userId.isin(counts[counts>=15].index)]
print('Number of users who have rated 25 or more items =', len(electronics_df1_final))
print('Number of unique users in the final data = ', electronics_df1_final['userId'].nunique())
print('Number of unique products in the final data = ', electronics_df1_final['userId'].nunique())


# In[69]:


#constructing the pivot table
final_ratings_matrix = electronics_df1_final.pivot(index = 'userId', columns ='productId', values = 'ratings').fillna(0)
final_ratings_matrix.head()


# In[70]:


print('Shape of final_ratings_matrix: ', final_ratings_matrix.shape)


# In[71]:


#Calucating the density of the rating marix
given_num_of_ratings = np.count_nonzero(final_ratings_matrix)
print('given_num_of_ratings = ', given_num_of_ratings)
possible_num_of_ratings = final_ratings_matrix.shape[0] * final_ratings_matrix.shape[1]
print('possible_num_of_ratings = ', possible_num_of_ratings)
density = (given_num_of_ratings/possible_num_of_ratings)
density *= 100
print ('density: {:4.2f}%'.format(density))


# # Splitting the data
# 

# In[72]:


#Split the data randomnly into train and test datasets into 70:30 ratio
train_data, test_data = train_test_split(electronics_df1_final, test_size = 0.3, random_state=0)
train_data.head()


# In[73]:



print('Shape of training data: ',train_data.shape)
print('Shape of testing data: ',test_data.shape)


# # Building the popularity model

# In[74]:


#Count of user_id for each unique product as recommendation score 
train_data_grouped = train_data.groupby('productId').agg({'userId': 'count'}).reset_index()
train_data_grouped.rename(columns = {'userId': 'score'},inplace=True)
train_data_grouped.head(40)


# In[75]:


#Sort the products on recommendation score 
train_data_sort = train_data_grouped.sort_values(['score', 'productId'], ascending = [0,1]) 
      
#Generate a recommendation rank based upon score 
train_data_sort['rank'] = train_data_sort['score'].rank(ascending=0, method='first') 
          
#Get the top 5 recommendations 
popularity_recommendations = train_data_sort.head(5) 
popularity_recommendations


# In[76]:



# Use popularity based recommender model to make predictions
def recommend(user_id):     
    user_recommendations = popularity_recommendations 
          
    #Add user_id column for which the recommendations are being generated 
    user_recommendations['userId'] = user_id 
      
    #Bring user_id column to the front 
    cols = user_recommendations.columns.tolist() 
    cols = cols[-1:] + cols[:-1] 
    user_recommendations = user_recommendations[cols] 
          
    return user_recommendations


# In[77]:


find_recom = [10,100,150]   # This list is user choice.
for i in find_recom:
    print("The list of recommendations for the userId: %d\n" %(i))
    print(recommend(i))    
    print("\n")


# # Building Collaborative Filtering recommender model

# In[78]:


electronics_df_CF = pd.concat([train_data, test_data]).reset_index()
electronics_df_CF.head()


# In[79]:


# Matrix with row per 'user' and column per 'item' 
pivot_df = electronics_df_CF.pivot(index = 'userId', columns ='productId', values = 'ratings').fillna(0)
pivot_df.head()


# In[80]:


print('Shape of the pivot table: ', pivot_df.shape)


# In[81]:


#define user index from 0 to 10
pivot_df['user_index'] = np.arange(0, pivot_df.shape[0], 1)
pivot_df.head()


# In[82]:


pivot_df.set_index(['user_index'], inplace=True)
# Actual ratings given by users
pivot_df.head()


# In[83]:


# Singular Value Decomposition
U, sigma, Vt = svds(pivot_df, k = 10)


# In[84]:


print('Left singular matrix: \n',U)


# In[85]:


print('Sigma: \n',sigma)


# In[86]:


# Construct diagonal array in SVD
sigma = np.diag(sigma)
print('Diagonal matrix: \n',sigma)


# In[87]:


print('Right singular matrix: \n',Vt)


# In[88]:


#Predicted ratings
all_user_predicted_ratings = np.dot(np.dot(U, sigma), Vt) 
# Convert predicted ratings to dataframe
preds_df = pd.DataFrame(all_user_predicted_ratings, columns = pivot_df.columns)
preds_df.head()


# In[89]:


# Recommend the items with the highest predicted ratings

def recommend_items(userID, pivot_df, preds_df, num_recommendations):
    # index starts at 0  
    user_idx = userID-1 
    # Get and sort the user's ratings
    sorted_user_ratings = pivot_df.iloc[user_idx].sort_values(ascending=False)
    #sorted_user_ratings
    sorted_user_predictions = preds_df.iloc[user_idx].sort_values(ascending=False)
    #sorted_user_predictions
    temp = pd.concat([sorted_user_ratings, sorted_user_predictions], axis=1)
    temp.index.name = 'Recommended Items'
    temp.columns = ['user_ratings', 'user_predictions']
    temp = temp.loc[temp.user_ratings == 0]   
    temp = temp.sort_values('user_predictions', ascending=False)
    print('\nBelow are the recommended items for user(user_id = {}):\n'.format(userID))
    print(temp.head(num_recommendations))


# In[90]:


userID = 4
num_recommendations = 5
recommend_items(userID, pivot_df, preds_df, num_recommendations)


# In[91]:


userID = 6
num_recommendations = 5
recommend_items(userID, pivot_df, preds_df, num_recommendations)


# In[92]:


userID = 4
num_recommendations = 5
recommend_items(userID, pivot_df, preds_df, num_recommendations)


# In[93]:


# Actual ratings given by the users
final_ratings_matrix.head()


# In[94]:


# Average ACTUAL rating for each item
final_ratings_matrix.mean().head()


# In[95]:


# Predicted ratings 
preds_df.head()


# In[96]:


# Average PREDICTED rating for each item
preds_df.mean().head()


# In[97]:


rmse_df = pd.concat([final_ratings_matrix.mean(), preds_df.mean()], axis=1)
rmse_df.columns = ['Avg_actual_ratings', 'Avg_predicted_ratings']
print(rmse_df.shape)
rmse_df['item_index'] = np.arange(0, rmse_df.shape[0], 1)
rmse_df.head()


# In[98]:


RMSE = round((((rmse_df.Avg_actual_ratings - rmse_df.Avg_predicted_ratings) ** 2).mean() ** 0.5), 5)
print('\nRMSE SVD Model = {} \n'.format(RMSE))


# In[99]:


# Enter 'userID' and 'num_recommendations' for the user #
userID = 9
num_recommendations = 5
recommend_items(userID, pivot_df, preds_df, num_recommendations)


# # Summary of the insight 

# 
# The Popularity-based recommender system is a non-personalised recommender system and these are based on frequecy counts, which may be not suitable to the user.We can see the differance above for the user id 4, 6 & 8, The Popularity based model has recommended the same set of 5 products to both but Collaborative Filtering based model has recommended entire different list based on the user past purchase history.
# 
# Model-based Collaborative Filtering is a personalised recommender system, the recommendations are based on the past behavior of the user and it is not dependent on any additional information.# 

# In[102]:


import anvil.server

anvil.server.connect("UFC4JCQC4VRSCTT52UZ54M3C-KLMKZAVZKRHTAPP4")


# In[ ]:




