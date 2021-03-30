"""
Exercise 3/Lab12
################

k-means++

Show that k-means++ provides better centroid initialisation by comparing the average value of  𝐽  
(i.e. the inertia_ attribute) for random initializations with k-means++ initialisations. More specifically:

- on the dataset d below, run the k-means algorithm 1000 times using random initialisation and record the inertia_ attribute. 
Plot the histogram of the recorded values and print their mean;
- repeat the process using k-means++;
- Which method performs better?

For the KMeans function, make sure to always use the parameters:
max_iter=1, n_init=1, algorithm='full', n_clusters=3, random_state=None
to emphasize the effect.

##########################
"""

from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import statistics
import pandas as pd
import matplotlib.pyplot as plt

n_samples = 1500
random_state = 100
X, y = make_blobs(n_samples=n_samples, random_state=random_state)
d = pd.DataFrame(X, columns=['X1', 'X2'])

# number of runs:
n_runs = 1000  # ----------------- to change in 1000 --------------------

# lists for keeping track of inertia_ attributes in k-means and k-means++:
inertia_kmeans = []
inertia_kmeansplus = []


# # function for counting number of distinct inertia_ attributes in histogram:
# def count():
#     dist_inertias_kmeans = []
#     dist_inertias_kmeansplus = []
#     for i in range(len(inertia_kmeans)):
#         if inertia_kmeans[i] in dist_inertias_kmeans:
#             pass
#         else:
#             dist_inertias_kmeans.append(inertia_kmeans[i])
#     for i in range(len(inertia_kmeansplus)):
#         if inertia_kmeansplus[i] in dist_inertias_kmeansplus:
#             pass
#         else:
#             dist_inertias_kmeansplus.append(inertia_kmeansplus[i])
#     return len(dist_inertias_kmeans), len(dist_inertias_kmeansplus)

"""
-------------------------- Parameters for KMeans -----------------------
- init controls the initialization technique. The standard version of the k-means algorithm is implemented by setting 
init to "random". Setting this to "k-means++" employs an advanced trick to speed up convergence, which you’ll use later.
- n_clusters sets k for the clustering step. This is the most important parameter for k-means.
- n_init sets the number of initializations to perform. This is important because two runs can converge on different 
cluster assignments. The default behavior for the scikit-learn algorithm is to perform ten k-means runs and return the 
results of the one with the lowest SSE.
- max_iter sets the number of maximum iterations for each initialization of the k-means algorithm.

-------------------------------- Attributes ----------------------------
- cluster_centers_ndarray of shape (n_clusters, n_features):
    Coordinates of cluster centers. If the algorithm stops before fully converging (see tol and max_iter), 
    these will not be consistent with labels_.
- labels_ndarray of shape (n_samples,):
    Labels of each point
- inertia_float:
    Sum of squared distances of samples to their closest cluster center.
- n_iter_int:
    Number of iterations run.
"""

kmeans = KMeans(init="random", n_clusters=3, n_init=1, max_iter=1, algorithm='full', random_state=None)
# clusters = kmeans.cluster_centers_

""" ################################################# K-Means ##################################################### """

# iterating for k-means:
for _ in range(n_runs):
    # creating k-means object
    kmeans = KMeans(init="random", n_clusters=3, n_init=1, max_iter=1, algorithm='full', random_state=None).fit(X)
    centroids = kmeans.cluster_centers_
    pred_y = kmeans.fit(X)
    y_km = kmeans.fit_predict(X)
    # appending inertia_ attribute to list:
    inertia_kmeans.append(kmeans.inertia_)

# (optional:) categorizing (and plotting) the data using the optimal number of clusters (3):
plt.scatter(X[:, 0], X[:, 1], c=kmeans.labels_.astype(float), s=50, alpha=0.5)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=100, c='red')
plt.title("Clusters (K-means)")
plt.show()

# count_1, count_2 = count()

# (mandatory:) plotting histogram for k-means inertia_ attribute:
plt.title('K-means inertia_ attribute')
plt.hist(inertia_kmeans, bins=10)
plt.show()
print("inertia_kmeans: ", inertia_kmeans)
mean_kmeans = statistics.mean(inertia_kmeans)
print(f'Mean for K-means: ', {mean_kmeans})

""" ################################################ K-Means++ #################################################### """

# iterating for k-means++:
kmeansplus = KMeans(init="k-means++", n_clusters=3, n_init=1, max_iter=1, algorithm='full', random_state=None)
for _ in range(n_runs):
    # creating k-means object
    kmeansplus = KMeans(init="k-means++", n_clusters=3, n_init=1, max_iter=1, algorithm='full', random_state=None)
    pred_y_plus = kmeansplus.fit(X)
    y_km = kmeansplus.fit_predict(X)
    inertia_kmeansplus.append(kmeans.inertia_)

# (optional:) categorizing (and plotting) the data using the optimal number of clusters (3):
plt.scatter(X[:, 0], X[:, 1], c=kmeansplus.labels_.astype(float), s=50, alpha=0.5)
plt.scatter(kmeansplus.cluster_centers_[:, 0], kmeansplus.cluster_centers_[:, 1], s=100, c='red')
plt.title("Clusters (K-means++)")
plt.show()

# (mandatory:) plotting histogram for k-means++ inertia_ attribute:
plt.title('K-means inertia_ attribute')
plt.hist(inertia_kmeansplus, bins=10)
plt.show()
mean_kmeansplus = statistics.mean(inertia_kmeansplus)
print("inertia_kmeans++: ", inertia_kmeansplus)
print(f'Mean for K-means++: ', {mean_kmeansplus})


""" 
#######################################################################################################################
K-means++ performs better because it doesn't pick the centroids randomly, but instead tries to spread them evenly across
the surface of the plot by generating the first one randomly and choosing the position of the following ones depending
on the first one. Thus K-means++ helps in the sense that it only finds one value for the inertia_ attribute. 
#######################################################################################################################
"""
