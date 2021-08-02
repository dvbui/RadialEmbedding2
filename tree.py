import math
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy as np
from radii import find_radii
from clustering import clustering
from find_ring import closest
from ensure_min_dist import ensure_min_dist
from python_tsp.heuristics import solve_tsp_simulated_annealing


def tsp_solver_dp(n, d):
    if n == 0:
        return (0,[])
    if n<=16:
        dp = [[None for i in range(1<<n)] for j in range(n)]
        trace = [[None for i in range(1<<n)] for j in range(n)]

        def caldp(x,y):
            if dp[x][y] is not None:
                return dp[x][y]
            if y==(1<<n)-1:
                dp[x][y] = d(x, 0)
            else:
                for i in range(n):
                    if ((y>>i) & 1) == 0:
                        tmp = caldp(i, y|(1<<i)) + d(x, i)
                        if dp[x][y] is None or tmp < dp[x][y]:
                            dp[x][y] = tmp
                            trace[x][y] = i
            return dp[x][y]
        
        caldp(0,1)
        current_state = (0,1)
        order = [0]
        for i in range(n-1):
            new_node = trace[current_state[0]][current_state[1]]
            order.append(new_node)
            current_state = (new_node, current_state[1]|(1<<new_node))
        return (caldp(0,1), order)

def tsp_solver_greedy(n, d):
    if n == 0:
        return (0,[])
    if n == 1:
        return (0,[0])
    if n == 2:
        return (d(0, 1)*2, [0,1])

    distance_matrix = lambda x,y: d(x, y)
    order = [0]
    used = {0: None}
    total_distance = 0
    current_node = 0
    for i in range(n-1):
        best_node = None
        best_dist = 1e9
        for j in range(n):
            if j in used:
                continue
            if distance_matrix(current_node,j) < best_dist:
                best_dist = distance_matrix(current_node,j)
                best_node = j
        assert(best_node is not None)
        total_distance += best_dist
        order.append(best_node)
        used[best_node] = None
    return (total_distance, order)

def tsp_solver(n, d):
    if n==0:
        return (0,[])
    if n<=16:
        return tsp_solver_dp(n,d)
    return tsp_solver_greedy(n,d)


def fit_transform(distance_matrix, min_dist=0, number_of_rings=4):
    focus = 0
    entities = list(range(1, len(distance_matrix)))
    d1 = lambda x,y: distance_matrix[x][y] if x!=y else 0
    radius = find_radii(focus, entities, d1, number_of_rings)

    seeds = [[] for i in range(number_of_rings)]
    
    v = [[] for i in range(len(distance_matrix))]
    parent = [-1 for i in range(len(distance_matrix))]
    first_ring_parent = [-1 for i in range(len(distance_matrix))]
    for i in entities:
        seeds[closest(d1(focus, i), radius)].append(i)
    
    # create a tree

    for i in seeds[0]:
        v[0].append(i)
        parent[i] = 0
        first_ring_parent[i] = i
    
    for i in range(1, number_of_rings):
        for x in seeds[i]:
            best_y = None
            best_d1 = 0
            for y in seeds[i-1]:
                if best_y is None or d1(x,y) < best_d1:
                    best_y = y
            v[y].append(x)
            parent[x] = y
            first_ring_parent[x] = first_ring_parent[y]
    
    order = [[] for i in range(number_of_rings)]

    tsp_length, tsp_order = tsp_solver(len(seeds[0]), lambda x,y: distance_matrix[seeds[0][x]][seeds[0][y]])
    order[0] = list(map(lambda x: seeds[0][x], tsp_order))

    angle_allowance = [distance_matrix[order[0][i]][order[0][i-1]] * 2 * math.pi / tsp_length for i in range(len(seeds[0]))]
    assert(abs(sum(angle_allowance)-2*math.pi) < 1e-6)


    ## Assumption made at this point: Each ring has at least one point
    for i in range(1, number_of_rings):
        best_first = None
        best_first_distance = None
        current = 0
        while best_first is None and current < len(order[i-1]):
            for j in v[order[i-1][current]]:
                if best_first is None or distance_matrix[j][order[i-1][current]] < best_first_distance:
                    best_first = j
                    best_first_distance = distance_matrix[j][order[i-1][current]]
        order[i].append(best_first)
        
        

        
        
        







        
    

    

    

