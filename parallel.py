import math
import numpy as np
from radii import *
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from find_ring import closest
from ensure_min_dist import ensure_min_dist

test_name = None
loop_iteration = -1
eps = 1e-10

small_first_ring = 0

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

"""
Find the order of consideration for the focus' neighboring points.
Algorithm:
- Choose an arbitrary point as the current point
- Choose the closest point to the current point such that it hasn't been chosen, is in the same group and is closest.
- If there is no same group and non-chosen point, choose the closest non-chosen point.
- Keep building the order until there is no point left.
"""
def entities_order(entities, d1):
    order = [0]
    chosen = set([0])
    for i in range(len(entities)-1):
        last_part = order[-1]
        best_neighbor = -1
        for j in range(len(entities)):
            if j in chosen or entities[j]["group"] != entities[last_part]["group"]:
                continue
            if best_neighbor == -1 or d1(entities[best_neighbor],entities[last_part]) > d1(entities[j], entities[last_part]):
                best_neighbor = j
        
        if best_neighbor == -1:
            for j in range(len(entities)):
                if j in chosen:
                    continue
                if best_neighbor == -1 or d1(entities[best_neighbor],entities[last_part]) > d1(entities[j], entities[last_part]):
                    best_neighbor = j
        
        order.append(best_neighbor)
        chosen.add(best_neighbor)
    
    return list(map(lambda x: entities[x], order))

def fit_transform(distance_matrix, min_dist=0, number_of_rings=4):
    data = clustering(distance_matrix)
    entities = data["data"]
    focus = data["focus"]
    d1 = lambda x,y: distance_matrix[x["object"]][y["object"]]
    order = entities_order(entities, d1)
    radii = find_radii(focus, entities, d1, number_of_rings)

    partition = [[None for j in range(101)] for i in range(len(radii))]
    position_counter = 0
    prev_group = -1

    """
    Put the points to the ring according to the greedy order found above.
    Imagine the rings as rows in the table, each position on a specific ring is a cell in that table.
    position_counter = column index
    We need to put the points in the correct row.
    If the cell on this row and at the position counter is empty, we could put the point there.
    If this is not the first point of the group, 
    we also consider 3 columns before that, and choose the one that would give the current point
    the closest neighbor if that neighbor is in the top 5)
    (Rationale for the above steps: Sometimes it's better to move some points back to save space.
    Since each column occupies an angle, the larger the number of columns, the closer the points
    and the harder to distinguish the columns.
    Rationale for the number 3: it helps the point to stay with its cluster
    Rationale for not considering the first point: since clusters are supposed to be far away
    from each other, the closest neighbor of the first point in other clusters might not be close).
    """

    for e in order:
        best_r = closest(d1(focus, e), radii)
        
        best_position = None
        
        if partition[best_r][position_counter] is not None:
            position_counter+=1
            best_position = position_counter
        elif prev_group != e["group"]:
            position_counter+=1
            best_position = position_counter
        else:
            
            best_dist = sorted([d1(e, x) for x in entities])[5]

            for t in reversed(range(max(0, position_counter-2), position_counter+1)):
                if partition[best_r][t] is not None:
                    break

                candidates = {}
                for j in range(len(radii)):
                    for z in range(position_counter+1):
                        if partition[j][z] != None:
                            tmp = abs(best_r - j) + abs(t-z)
                            if tmp not in candidates:
                                candidates[tmp] = []
                            candidates[tmp].append(d1(partition[j][z], e))
                
                for k in sorted(candidates.keys()):
                    if best_position is None or sum(candidates[k])/len(candidates[k]) < best_dist:
                        best_position = t
                        best_dist = sum(candidates[k])/len(candidates[k])
                    break
                    
            if best_position is None:
                best_position = position_counter
        
        partition[best_r][best_position] = e
        
        prev_group = e["group"]

    """
    See: onering.py
    """
    
    parts = []
    for p in range(position_counter):
        dist = -1
        for i in range(len(partition)):
            for j in range(len(partition)):
                if partition[i][p] is None or partition[j][p+1] is None:
                    continue
                if i==j and partition[i][p]["group"]!=partition[j][p+1]["group"]:
                    dist = 2
                else:
                    dist = max(dist, d1(partition[i][p], partition[j][p+1]))
        
        if dist == -1:
            dist = 2
        parts.append(dist)

    # default last distance = max distance to signify that the last points in the TSP-greedy order 
    # might not be close to the first one = 1
    # This needs to be changed if cosine similarity is not used.
    parts.append(2)

    prefix = [parts[0]]
    for i in range(1,len(parts)):
        prefix.append(prefix[-1]+parts[i])
    total_len = sum(parts)
    assert(abs(total_len-prefix[-1])<1e-6)

    angle = [0] + list(map(lambda x: x/total_len*math.pi*2, prefix))
    result = [[focus, (0,0)]]
    for i in range(len(partition)):
        prev = None
        r = 0.1*(2 ** (i+1))
        for j in range(position_counter+1):
            if partition[i][j] is not None:
                actual_angle = angle[j]
                new_pos = (r * math.cos(actual_angle), r * math.sin(actual_angle))
                prev = actual_angle
                result.append([partition[i][j], new_pos])
    
    result.sort(key=lambda x: x[0]["object"])
    return ensure_min_dist(list(map(lambda x: [x[1][0], x[1][1]], result)), min_dist)

            

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