import math
import json
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
data_file_suffix = ""
with open('./data/abstracts' + data_file_suffix + '.json') as file:
  abstracts = json.load(file)

tfidfVectorizer = TfidfVectorizer()
tfidfvectors = tfidfVectorizer.fit_transform(abstracts)
tfidf_top100 = np.load('./data/tfidfDist' + data_file_suffix + '.npy')

distance_cache = {}

def d(x,y):
    len1 = math.sqrt((x * x.transpose()).toarray()[0][0])
    len2 = math.sqrt((y * y.transpose()).toarray()[0][0])
    num = abs((x * y.transpose()).toarray()[0][0])
    return max(0,1 - num / (len1*len2))

def d_cosine(x, y, is_estimate=False):
    if x==y:
        return 0
    if (x,y) in distance_cache:
        return distance_cache[(x,y)]
    
    for i in range(len(tfidf_top100[0][x])):
        if abs(tfidf_top100[1][x][i] - y) < 1e-6:
            distance_cache[(x,y)] = tfidf_top100[0][x][i]
            distance_cache[(y,x)] = tfidf_top100[0][x][i]
            return tfidf_top100[0][x][i]
    for i in range(len(tfidf_top100[0][y])):
        if abs(tfidf_top100[1][y][i] - x) < 1e-6:
            distance_cache[(x,y)] = tfidf_top100[0][y][i]
            distance_cache[(y,x)] = tfidf_top100[0][y][i]
            return tfidf_top100[0][y][i]
    if not is_estimate:
        distance_cache[(x,y)] = d(tfidfvectors[x], tfidfvectors[y])
    else:
        distance_cache[(x,y)] = 0.5 + 0.5*max(tfidf_top100[0][y][-1], tfidf_top100[0][x][-1])
    
    distance_cache[(y,x)] = distance_cache[(x,y)]
    return distance_cache[(x,y)]

def create_distance_matrix(id,is_estimated=False):
    neighbor_ids = [int(id)]
    for index in range(1, 101):
        neighbor_id = int(tfidf_top100[1][int(id)][index])
        neighbor_ids.append(neighbor_id)
    distance_matrix = [[d_cosine(neighbor_ids[i], neighbor_ids[j], is_estimated) for i in range(101)] for j in range(101)]
    return distance_matrix

def d2(x,y):
    r1 = math.hypot(x[0], x[1])
    r2 = math.hypot(y[0], y[1])
    if r1 > r2:
        r1, r2 = r2, r1
        x, y = y , x
    phi1 = math.atan2(x[1], x[0])
    phi2 = math.atan2(y[1], y[0])

    diff_angle = abs(phi1 - phi2)
    diff_angle %= (2*math.pi)
    diff_angle = min(diff_angle, 2*math.pi - diff_angle)
    return (r2 - r1) + r1 * diff_angle