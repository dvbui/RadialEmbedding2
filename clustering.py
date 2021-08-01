from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy as np

def clustering(distance_matrix):
    data = {}
    data["focus"] = {}
    data["focus"]["object"] = 0
    data["data"] = []
    for i in range(1, len(distance_matrix)):
        data["data"].append({"object": i})
    
    X = np.array([distance_matrix[i][1:] for i in range(1, len(distance_matrix))])
    best_value = -1e9
    best_k = 2
    for k in range(2, min(len(distance_matrix)-1, 9)):
        clustering = KMeans(random_state=0,n_clusters=k).fit(X)
        labels = clustering.labels_
        tmp = silhouette_score(X, labels, metric="precomputed")
        if tmp > best_value:
            best_value = tmp
            best_k = k
    clustering = KMeans(random_state=0, n_clusters = best_k).fit(X)
    labels = list(clustering.labels_.tolist())
    data["focus"]["group"] = best_k
    for i in range(len(data["data"])):
        data["data"][i]["group"] = labels[i]
    return data

