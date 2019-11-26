## Import libraries
import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score,calinski_harabasz_score

## Load data
dataset = pd.read_json('Brisbane_CityBike.json','r',encoding='utf-8')


## Extract 'latitude' and 'longitude' as features
X = dataset.iloc[:,3:5]


## Find the optimal K and build KMeans model

# use "elbow function" , calinski_harabasz_score and silhouette_score to choose K
distortions = []
silhouette_scores = []
calinski_harabasz_scores = []
kRange=range(2,10)
for k in kRange:
    model = KMeans(n_clusters=k, random_state=9)
    y_pred = model.fit_predict(X)
    distortion=sum(np.min(cdist(X, model.cluster_centers_, 'euclidean'), axis=1)) / X.shape[0]
    distortions.append(distortion)
    calinski_harabasz_scores.append(calinski_harabasz_score(X, y_pred))
    silhouette_scores.append(silhouette_score(X,y_pred))

#plot the distortions
fig = plt.figure(0)
plt.plot(kRange,distortions)
plt.xlabel('K')
plt.ylabel('distortion')
plt.savefig("elbow_function.png") #save the metric curve
plt. close(0)
# very hard to find the "elbow", maybe k = 3, 4 or 6 is optimal


#plot the calinski_harabasz_scores
fig = plt.figure(1)
plt.plot(kRange,calinski_harabasz_scores)
plt.xlabel('K')
plt.ylabel('calinski_harabasz_score')
plt.savefig("calinski_harabasz_score.png") #save the metric curve
plt. close(1)
# calinski_harabasz_score: the bigger the better, k = 3 or 6 is a local optimal

#plot the silhouette_score
fig = plt.figure(2)
plt.plot(kRange,silhouette_scores)
plt.xlabel('K')
plt.ylabel('silhouette_score')
plt.savefig("silhouette_score.png") #save the metric curve
plt. close(2)
# silhouette_score: the bigger the better, k = 6 is a local optimal

# choose k = 6 by considering the 3 metrics
model = KMeans(n_clusters=6, random_state=9)
y_pred = model.fit_predict(X)

# plot the clustering result with 2 features
fig = plt.figure(3)
plt.scatter(X['latitude'], X['longitude'], c=y_pred)
plt.xlabel('latitude')
plt.ylabel('longitude')
plt.savefig("result2D.png") #save clustering result
plt. close(3)

# Extract 'number' as feature
X = dataset['number']
X = X.to_numpy()
X = X.reshape(-1,1)

# Build KMeans model
# clustering 1D data might not be meaningful
# but we can still try to see the result
kmeans = KMeans(n_clusters=5, random_state=9)
y_pred = kmeans.fit_predict(X)

# plot the clustering result with 1 feature
fig = plt.figure(4)
plt.scatter(X,y_pred, c=y_pred)
plt.xlabel('class')
plt.ylabel('number')
plt. close(4)
# we can see a extreme sample here

unique,counts=np.unique(kmeans.labels_,return_counts=True)
dict(zip(unique,counts))
# take a look for how many samples in each class
# there is only one bike station in class 1

# clustering result for k = 2
kmeans = KMeans(n_clusters=2, random_state=9)
y_pred = kmeans.fit_predict(X)
fig = plt.figure(5)
plt.scatter(X,y_pred, c=y_pred)
plt.xlabel('class')
plt.ylabel('number')
plt.savefig("result1D.png")
plt. close(5)
# So the conclusion is that k = 2 is optimal for 1D clustering in "number"
# For the extreme big station, we can call it 'Super Station'
# For the rest, we can call them 'Normal Station'