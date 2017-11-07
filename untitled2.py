#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 14:58:09 2017

@author: Alamgir
"""

# Import libraries necessary for this project

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display # Allows the use of display() for DataFrames

# Import supplementary visualizations code visuals.py
import visuals as vs

# Pretty display for notebooks
#%matplotlib inline

# Load the wholesale customers dataset
try:
    data = pd.read_csv("customers.csv")
    data.drop(['Region', 'Channel'], axis = 1, inplace = True)
    print("Wholesale customers dataset has {} samples with {} features each.".format(*data.shape))
except:
    print("Dataset could not be loaded. Is the dataset missing?")
    
    
####################
# Display a description of the dataset
display(data.describe())

######################### 
# TODO: Select three indices of your choice you wish to sample from the dataset
indices = [2,7,14]

######################################
# Create a DataFrame of the chosen samples
samples = pd.DataFrame(data.loc[indices], columns = data.keys()).reset_index(drop = True)
print ("Chosen samples of wholesale customers dataset:")
display(samples)

# TODO: Make a copy of the DataFrame, using the 'drop' function to drop the given feature
# Lets drop Detergents_paper
new_data = pd.DataFrame(data.drop(['Detergents_Paper'], axis=1))
target   = data['Detergents_Paper']
# TODO: Split the data into training and testing sets(0.25) using the given feature as the target
# Set a random state.
from sklearn.model_selection import  train_test_split
X_train, X_test, y_train, y_test = train_test_split(new_data, target, test_size = 0.25)

# TODO: Create a decision tree regressor and fit it to the training set
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor()
regressor.fit(X_train, y_train)

# TODO: Report the score of the prediction using the testing set
score = regressor.score(X_test, y_test)
print(score)

##################################
# Produce a scatter matrix for each pair of features in the data
pd.scatter_matrix(data, alpha = 0.3, figsize = (14,8), diagonal = 'kde');
#pd.plotting.scatter_matrix(data, alpha = 0.3, figsize = (14,8), diagonal = 'kde');

##############################
# TODO: Scale the data using the natural logarithm
log_data = np.log(data)

# TODO: Scale the sample data using the natural logarithm
log_samples = np.log(samples)

# Produce a scatter matrix for each pair of newly-transformed features
pd.scatter_matrix(log_data, alpha = 0.3, figsize = (14,8), diagonal = 'kde');


#####################
outliers  = np.zeros(log_data.shape[0])
# For each feature find the data points with extreme high or low values
for feature in log_data.keys():
    
    # TODO: Calculate Q1 (25th percentile of the data) for the given feature
    Q1 = np.percentile(log_data[feature], 25)
    
    # TODO: Calculate Q3 (75th percentile of the data) for the given feature
    Q3 = np.percentile(log_data[feature], 75)
    
    # TODO: Use the interquartile range to calculate an outlier step (1.5 times the interquartile range)
    step = 1.5*(Q3-Q1)
    
    # Display the outliers
    print("Data points considered outliers for the feature '{}':".format(feature))
    outlier = ~((log_data[feature] >= Q1 - step) & (log_data[feature] <= Q3 + step))
    display(log_data[outlier])
    
    #accumulate outliers for removal
    outliers = outliers + outlier.astype(int)  
    
# OPTIONAL: Select the indices for data points you wish to remove
#outliers  = list(set(outliers)) #reove duplicate indices

# Remove the outliers, if any were specified
# lets remove if the at least two features indicate so
good_data = log_data[outliers < 2]
#pd.scatter_matrix(good_data, alpha = 0.3, figsize = (14,8), diagonal = 'kde');
#print('Outliers :'+ str(sum(outliers != 0)) )
#print(len(good_data))
#good_data.head(10)
#sum(outliers > 1)

###############################
# TODO: Apply PCA by fitting the good data with the same number of dimensions as features
from sklearn.decomposition import PCA
pca = PCA()
pca.fit(good_data)

# TODO: Transform log_samples using the PCA fit above
pca_samples = pca.transform(log_samples)

# Generate PCA results plot
pca_results = vs.pca_results(good_data, pca)



###############################
# Display sample log-data after having a PCA transformation applied
display(pd.DataFrame(np.round(pca_samples, 4), columns = pca_results.index.values))



# TODO: Apply PCA by fitting the good data with only two dimensions
pca = PCA(2).fit(good_data)

# TODO: Transform the good data using the PCA fit above
reduced_data = pca.transform(good_data)

# TODO: Transform log_samples using the PCA fit above
pca_samples = pca.transform(log_samples)

# Create a DataFrame for the reduced data
reduced_data = pd.DataFrame(reduced_data, columns = ['Dimension 1', 'Dimension 2'])


# Display sample log-data after applying PCA transformation in two dimensions
display(pd.DataFrame(np.round(pca_samples, 4), columns = ['Dimension 1', 'Dimension 2']))


# Create a biplot
vs.biplot(good_data, reduced_data, pca)



# TODO: Apply your clustering algorithm of choice to the reduced data 
#from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
#clusterer = GaussianMixture().fit(reduced_data)
clusterer = KMeans(n_clusters=8, random_state=0).fit(reduced_data)


# TODO: Predict the cluster for each data point
preds = clusterer.predict(reduced_data)

#print(preds)
# TODO: Find the cluster centers
centers = clusterer.cluster_centers_
#print(len(centers))
# TODO: Predict the cluster for each transformed sample data point
sample_preds = clusterer.predict(pca_samples)

# TODO: Calculate the mean silhouette coefficient for the number of clusters chosen
from sklearn.metrics import silhouette_score
score = silhouette_score(reduced_data, preds, random_state=0)

print(score)



# Display the results of the clustering from implementation
vs.cluster_results(reduced_data, preds, centers, pca_samples)

# TODO: Inverse transform the centers
log_centers = pca.inverse_transform(centers)

# TODO: Exponentiate the centers
true_centers = np.exp(log_centers)

# Display the true centers
segments = ['Segment {}'.format(i) for i in range(0,len(centers))]
true_centers = pd.DataFrame(np.round(true_centers), columns = data.keys())
true_centers.index = segments
display(true_centers)
print(np.mean(data, axis=0))


# Display the predictions
for i, pred in enumerate(sample_preds):
    print("Sample point", i, "predicted to be in Cluster", pred)
    
# print samples too for convenience of answering Q9
samples


# Display the clustering results based on 'Channel' data
# need to adjust outliers from count values to index
outliers_idx = np.where(outliers > 0)
vs.channel_results(reduced_data, outliers_idx, pca_samples)
