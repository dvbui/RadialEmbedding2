import math
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy as np
from radii import find_radii
from clustering import clustering
from find_ring import closest
from ensure_min_dist import ensure_min_dist


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

    partition = [{} for i in range(len(radii))]
    
    """
    Put the points to the ring according to the greedy order found above.
    Imagine the rings as rows in the table, each position on a specific ring is a cell in that table.
    position_counter = column index
    We need to put the points in the correct row.
    If the cell on this row and at the position counter is empty, put the point there,
    otherwise increase the position counter, and put the point there.
    This works because consecutive points in the TSP order tend to be close to each other,
    so they should also be close in the layout.
    """
    position_counter = 0
    for e in order:
        best_r = closest(d1(focus, e), radii)
        if position_counter in partition[best_r]:
            position_counter+=1
        partition[best_r][position_counter] = e
    
    """
    Points in the same column will be at the same angle in the radial layout.
    The following code determines the distance between two columns on the table.
    Total distance should be 2*pi
    """

    """
    The distance between two columns is proportional to the distance magnitude of two neighboring points
    (same row, neighbor column).
    These two points are guaranteed to exist, except from the last column.
    """
    parts = []
    for p in range(position_counter):
        for i in range(len(partition)):
            if p in partition[i] and p+1 in partition[i]:
                #print(partition[i][p], partition[i][p+1])
                parts.append(d1(partition[i][p], partition[i][p+1]))
                break
    # default last distance = max distance to signify that the last points in the TSP-greedy order 
    # might not be close to the first one = 1
    # This needs to be changed if cosine similarity is not used.
    parts.append(1)

    
    prefix = [parts[0]]
    for i in range(1,len(parts)):
        prefix.append(prefix[-1]+parts[i])
    total_len = sum(parts)
    assert(abs(total_len-prefix[-1])<1e-6)


    """
    Generate the layout. At this point, we have the distance from the article to the focus, and the angle.
    We can easily calculate the position of the point on the layout.
    """
    angle = [0] + list(map(lambda x: x/total_len*math.pi*2, prefix))
    result = [[focus, (0,0)]]
    for i in range(len(partition)):
        prev = None
        r = 0.1*(2 ** (i+1))
        for j in range(position_counter+1):
            if j in partition[i]:
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