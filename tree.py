import math
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy as np
from radii import find_radii
from clustering import clustering
from find_ring import closest
from ensure_min_dist import ensure_min_dist





def fit_transform(distance_matrix, min_dist=0, number_of_rings=4):
    for i in range(1, len(distance_matrix)):
        