import math
from onering import entities_order
from radii import *
from clustering import clustering
from find_ring import closest
from ensure_min_dist import ensure_min_dist


def dfs(adj, x):
    result = [x]
    for j in adj[x]:
        result.extend(dfs(adj, j))
    return result

def entities_order(distance_matrix, seeds):
    v = [[] for i in range(len(distance_matrix))]
    for i in range(len(seeds)):
        for j in seeds[i]:
            if (i==0):
                v[0].append(j)
            else:
                best_parent = None
                best_parent_dist = None
                for z in seeds[i-1]:   
                    if best_parent is None or best_parent_dist < distance_matrix[z][j]:
                        best_parent = z
                        best_parent_dist = distance_matrix[z][j]
                v[best_parent].append(j)
    
    return dfs(v, 0)[1:]

def fit_transform(distance_matrix, min_dist=0, number_of_rings=4):
    n = len(distance_matrix)
    d1 = lambda x,y: distance_matrix[x][y]
    radii = find_radii(0, list(range(1, n)), d1, number_of_rings)
    seeds = [[] for i in range(number_of_rings)]
    for i in range(1, n):
        seeds[closest(d1(0, i), radii)].append(i)
    
    order = entities_order(distance_matrix, seeds)
    partition = [{} for i in range(number_of_rings)]

    current_column = 0
    for e in order:
        r = closest(d1(0, e), radii)
        top5 = sorted(distance_matrix[e])[5]
        same_column = []
        for i in range(number_of_rings):
            if current_column in partition[i]:
                same_column.extend(partition[i][current_column])
        
        flag = False
        for v in same_column:
            if d1(e,v) <= top5:
                flag = True
        
        if not (flag or len(same_column)==0):
            current_column+=1
        
        if current_column not in partition[r]:
            partition[r][current_column] = []
        partition[r][current_column].append(e)

    print(distance_matrix)
    # distance between column
    column_distances = []
    for i in range(current_column+1):
        A_list = []
        for j in range(number_of_rings):
            if i in partition[j]:
                A_list.append(partition[j][i][-1])
        B_list = []
        for j in range(number_of_rings):
            if (i+1)%(current_column+1) in partition[j]:
                B_list.append(partition[j][(i+1)%(current_column+1)][0])
        
        max_dist = 0
        for a in A_list:
            for b in B_list:
                max_dist = max(max_dist, distance_matrix[a][b])
        
        column_distances.append(max_dist)
    
    print(column_distances)
    sum_dist = sum(column_distances)
    column_angles = [column_distances[i] / sum_dist * 2 * math.pi for i in range(len(column_distances))]

    unit_distance = None
    for r in range(number_of_rings):
        for c in partition[r]:
            total_actual_dist = sum([distance_matrix[partition[r][c][i]][partition[r][c][i+1]] for i in range(len(partition[r][c])-1)]) + 1
            total_layout_dist = column_angles[c] * radii[r]
            if unit_distance is None or total_layout_dist / total_actual_dist < unit_distance:
                unit_distance = total_layout_dist / total_actual_dist
    
    current_angle = 0
    result = [[] for i in range(n)]

    for c in range(current_column+1):
        for r in range(number_of_rings):
            if c in partition[r]:
                current_angle_cell = current_angle
                unit_angle = unit_distance / radii[r]
                for i in range(len(partition[r][c])):
                    x = partition[r][c][i]
                    result[x] = [radii[r] * math.cos(current_angle_cell), radii[r] * math.sin(current_angle_cell)]
                    if i+1 < len(partition[r][c]):
                        current_angle_cell += unit_angle * distance_matrix[partition[r][c][i]][partition[r][c][i+1]]
        current_angle += column_angles[c]
    
    result[0] = [0,0]
    result = ensure_min_dist(result, min_dist)
    return result





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