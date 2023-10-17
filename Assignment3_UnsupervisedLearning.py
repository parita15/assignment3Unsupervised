#!/usr/bin/env python
# coding: utf-8

# In[2]:


from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.svm import SVC  
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import pairwise_distances
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score


# In[3]:


#This will fetch the olivetti_faces dataset
olivetti_faces = fetch_olivetti_faces(shuffle=True, random_state=42)
X, Y = olivetti_faces.data, olivetti_faces.target


# In[4]:


# Split the data into training and temporary sets (80% training, 20% temp)
X_train, X_temp, y_train, y_temp = train_test_split(
    olivetti_faces.data, olivetti_faces.target, test_size=0.2, random_state=42, stratify=olivetti_faces.target
)

# Split the temporary set into validation and test sets (50% validation, 50% test)
X_valid, X_test, y_valid, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)


# In[5]:


# This will define the support vector machine classifier
classifier = SVC(kernel='linear', C=1)

# The below will perform the k-fold cross-validation on the training set
k_fold = 5  
cv_scores = cross_val_score(classifier, X_train, y_train, cv=k_fold)

classifier.fit(X_train, y_train)

# The below will predict the classifier on the validation set
y_pred = classifier.predict(X_valid)

# Calculate accuracy on the validation set
validation_accuracy = accuracy_score(y_valid, y_pred)

# Print the cross-validation scores and validation accuracy
print(f"Cross-validation scores: {cv_scores}")
print(f"Validation accuracy: {validation_accuracy}")


# In[6]:


n_clusters = 10

# a) Euclidean Distance (Use for continuous data)
euclidean_distances = pairwise_distances(X, metric='euclidean')
model_euclidean = AgglomerativeClustering(n_clusters=n_clusters, linkage='average', affinity='precomputed')
labels_euclidean = model_euclidean.fit(euclidean_distances)

# b) Minkowski Distance (Find distance between 2 points)
p = 2
minkowski_distances = pairwise_distances(X, metric='minkowski', p=p)
model_minkowski = AgglomerativeClustering(n_clusters=n_clusters, linkage='average', affinity='precomputed')
model_minkowski.fit(minkowski_distances)

# c) Cosine Similarity (Find the distance between 2 vectords)
cosine_similarities = pairwise_distances(X, metric='cosine')
model_cosine = AgglomerativeClustering(n_clusters=n_clusters, linkage='average', affinity='precomputed')
model_cosine.fit(cosine_similarities)

#Graph for Euclidean Distance
plt.figure(figsize=(12, 4))
plt.subplot(131)
plt.title("Euclidean Distance")
plt.scatter(X[:, 0], X[:, 1], c=model_euclidean.labels_, cmap='rainbow')

#Graph for Minkowski Distance
plt.subplot(132)
plt.title(f"Minkowski Distance (p={p})")
plt.scatter(X[:, 0], X[:, 1], c=model_minkowski.labels_, cmap='rainbow')

#Graph for cosine similarity
plt.subplot(133)
plt.title("Cosine Similarity")
plt.scatter(X[:, 0], X[:, 1], c=model_cosine.labels_, cmap='rainbow')

plt.show()


# In[7]:


# List to store silhouette scores for different numbers of clusters
silhouette_scores = []

# Define a range of cluster numbers to try
cluster_range = range(2, 21)  # You can adjust the range as needed

for n_clusters in cluster_range:
    #Euclidean Distance
    model_euclidean = AgglomerativeClustering(n_clusters=n_clusters, linkage='average', affinity='precomputed')
    labels_euclidean = model_euclidean.fit_predict(euclidean_distances)
    silhouette_avg_euclidean = silhouette_score(euclidean_distances, labels_euclidean)
    silhouette_scores.append(silhouette_avg_euclidean)

# Find the number of clusters with the highest silhouette score for euclidean distance
best_n_clusters_euclidean = cluster_range[np.argmax(silhouette_scores)]
best_silhouette_score_euclidean = max(silhouette_scores)

print(f"Optimal number of clusters for euclidean distance: {best_n_clusters_euclidean}")
print(f"Best silhouette score for euclidean distance: {best_silhouette_score_euclidean}")


# In[8]:


# List to store silhouette scores for different numbers of clusters
silhouette_scores = []

# Define a range of cluster numbers to try
cluster_range = range(2, 21)  # You can adjust the range as needed

for n_clusters in cluster_range:
    #Minkowski Distance
    model_minkowski = AgglomerativeClustering(n_clusters=n_clusters, linkage='average', affinity='precomputed')
    labels_minkowski = model_minkowski.fit_predict(minkowski_distances)
    silhouette_avg_minkowski = silhouette_score(minkowski_distances, labels_minkowski)
    silhouette_scores.append(silhouette_avg_minkowski)

# Find the number of clusters with the highest silhouette score for minkowski distance
best_n_clusters_minkowski = cluster_range[np.argmax(silhouette_scores)]
best_silhouette_score_minkowski = max(silhouette_scores)

print(f"Optimal number of clusters for minkowski distance: {best_n_clusters_minkowski}")
print(f"Best silhouette score for minkowski distance: {best_silhouette_score_minkowski}")


# In[9]:


# List to store silhouette scores for different numbers of clusters
silhouette_scores = []

# Define a range of cluster numbers to try
cluster_range = range(2, 21)  # You can adjust the range as needed

for n_clusters in cluster_range:
    #Cosine Similarities
    model_cosine = AgglomerativeClustering(n_clusters=n_clusters, linkage='average', affinity='precomputed')
    labels_cosine = model_cosine.fit_predict(cosine_similarities)
    silhouette_avg_cosine = silhouette_score(cosine_similarities, labels_cosine)
    silhouette_scores.append(silhouette_avg_cosine)

# Find the number of clusters with the highest silhouette score for cosine similarities
best_n_clusters_cosine = cluster_range[np.argmax(silhouette_scores)]
best_silhouette_score_cosine = max(silhouette_scores)

print(f"Optimal number of clusters for cosine similarities: {best_n_clusters_cosine}")
print(f"Best silhouette score for cosine similarities: {best_silhouette_score_cosine}")


# In[ ]:




