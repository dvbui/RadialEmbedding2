import math
import itertools
import random
import pprint
import numpy as np
from python_tsp.heuristics import solve_tsp_simulated_annealing
from radii import find_radii

from timeit import default_timer as timer
from distance import d2
from find_ring import closest
from ensure_min_dist import ensure_min_dist

test_name = None
loop_iteration = -1
eps = 1e-10



# partition a set of entities based on a given set of radii
# return a list of lists
# O(nlog(g))
def partition(focus, entities, radius, d):
    radius.sort()
    result = [ [] for i in range(0, len(radius)) ]
    for e in entities:
        dist = d(focus, e)
        id = closest(dist, radius)
        result[id].append(e)
    return result

def tsp_solver_brute_force(entities, d):
    n = len(entities)
    if n==0:
        return (0,[])
    min_total = None
    best_path = None
    for perm in itertools.permutations(range(n)):
        current_total = 0
        for i in range(n):
            current_total += d(entities[perm[i]], entities[perm[(i+1)%n]])
        if min_total is None or current_total < min_total:
            min_total = current_total
            best_path = perm
    return (min_total,best_path)

# O(2^(n/g) * (n/g))
def tsp_solver_dp(entities, d):
    n = len(entities)

    if n == 0:
        return (0,[])
    
    dp = [[None for i in range(1<<n)] for j in range(n)]

    trace = [[None for i in range(1<<n)] for j in range(n)]


    def caldp(x,y):
        if dp[x][y] is not None:
            return dp[x][y]
        if y==(1<<n)-1:
            dp[x][y] = d(entities[x], entities[0])
        else:
            for i in range(n):
                if ((y>>i) & 1) == 0:
                    tmp = caldp(i, y|(1<<i)) + d(entities[x], entities[i])
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

def tsp_solver_sa(entities, d):
    n = len(entities)

    if n == 0:
        return (0,[])
    if n == 1:
        return (0,[0])
    if n == 2:
        return (d(entities[0], entities[1])*2, [0,1])
    
    distance_matrix = np.array([[d(entities[i],entities[j]) for i in range(n)] for j in range(n)])

    result = solve_tsp_simulated_annealing(distance_matrix)
    return (result[1], result[0])

def tsp_solver_greedy(entities, d):
    n = len(entities)
    
    if n == 0:
        return (0,[])
    if n == 1:
        return (0,[0])
    if n == 2:
        return (d(entities[0], entities[1])*2, [0,1])

    distance_matrix = lambda x,y: d(entities[x], entities[y])
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

def embed_to_circle(entities, d):
    order = tsp_solver_greedy(entities, d)[1]
    result = []
    for i in order:
        result.append(entities[i])
    return result


def distortion(x,y,fx,fy,d1,d2):
    p1 = d1(x,y)
    p2 = d2(fx,fy)
    return max(p2/p1,p1/p2)

# O(n)
def additional_distortion(groups, d1, d2):
    new_group = groups[-1]
    groups = list(itertools.chain.from_iterable(groups[:-1]))
    total = 0 
    for e in new_group:
        for v in groups:
            total += distortion(e[0], v[0], e[1], v[1], d1, d2)
    return total

# O(n^2)
def total_distortion(positions, d1, d2):
    res = 0
    everything = list(itertools.chain.from_iterable(positions))
    for i in range(len(everything)):
        for j in range(i+1,len(everything)):
            res += distortion(everything[i][0], everything[j][0], everything[i][1], everything[j][1], d1, d2)
    return res

# O(20*n^2)
def rotational_align(groups, radius, d1, d2):
    n = len(groups)
    result = []
    for i in range(n):
        m = len(groups[i])
        
        total_dist = 0
        for j in range(m):
            total_dist += d1(groups[i][j], groups[i][(j+1)%m])
        
        best_value = None
        best_new_result = None
        number_of_angles = 20
        for j in range(number_of_angles):
            angle = 2*math.pi/number_of_angles*(j+random.random())
            positions = []
            for j in range(m):
                new_point = (radius[i] * math.cos(angle), radius[i] * math.sin(angle))
                positions.append( (groups[i][j], new_point))
                if total_dist != 0:
                    angle += 2 * math.pi * d1(groups[i][j], groups[i][(j+1)%m]) / total_dist

            possible_new_result = result + [positions]
            current_value = additional_distortion(possible_new_result, d1, d2)
            if best_value is None or current_value < best_value:
                best_value = current_value
                best_new_result = possible_new_result
        
        result = best_new_result
    
    return result

#O(n^2)
def is_non_contracting(positions, d1, d2):
    everything = list(itertools.chain.from_iterable(positions))
    for i in range(len(everything)):
        for j in range(i+1, len(everything)):
            a = everything[i]
            b = everything[j]
            if d2(a[1], b[1]) < d1(a[0], b[0]):
                print(i, j, a[0], b[0], a[1], b[1])
                return False
    return True

# O(n^2)
def ensure_non_contracting(positions, d1, d2):
    everything = list(itertools.chain.from_iterable(positions))
    multiplier = 1.0
    for i in range(len(everything)):
        for j in range(i+1,len(everything)):
            a = everything[i]
            b = everything[j]
            if d2(a[1], b[1]) < d1(a[0], b[0]):
                multiplier = max(multiplier, d1(a[0], b[0]) / d2(a[1], b[1]))
    
    multiplier+=1e-4
    for i in range(len(positions)):
        for j in range(len(positions[i])):
            positions[i][j] = (positions[i][j][0], (positions[i][j][1][0] * multiplier, positions[i][j][1][1] * multiplier))
    
    assert(is_non_contracting(positions, d1, d2))
    return positions


adjustments = 0
critical_points = {}

def add_point(x):
    global critical_points
    x%=(2*math.pi)
    while x < 0:
        x+=2*math.pi
    while x>=2*math.pi:
        x-=2*math.pi
    critical_points[x] = []

def add_event(x, event):
    global critical_points
    x%=(2*math.pi)
    while x < 0:
        x+=2*math.pi
    while x>=2*math.pi:
        x-=2*math.pi
    if x in critical_points:
        critical_points[x].append(event)
    else:
        critical_points[x] = [event]

def same_angle(angle1, angle2):
    return (angle1-angle2)%(math.pi*2) < eps

# O(n)
def generate_critical_points(i,j,positions,d1,d2):
    global critical_points
    critical_points = {}
    everything = list(itertools.chain.from_iterable(positions))
    total_length = sum([len(positions[i]) for i in range(len(positions))])
    assert(len(everything)==total_length)
    def min_angle_diff(obj1, obj2):
        rx = math.hypot(obj1[1][0], obj1[1][1])
        ry = math.hypot(obj2[1][0], obj2[1][1])
        d = d1(obj1[0], obj2[0])
        return max(0, (d - abs(rx-ry))/min(rx,ry) + 1e-6)

    # generate critical_points
    add_point(0)
    for e in everything:
        angle = math.atan2(e[1][1], e[1][0])
        if d1(e[0], positions[i][j][0]) > 1e-6:
            angle_difference = min_angle_diff(positions[i][j], e)
            assert(angle_difference <= math.pi + 1e-6)
            angle_difference = min(angle_difference, math.pi)
            angle = math.atan2(e[1][1], e[1][0])
            add_point(angle - angle_difference)
            add_point(angle + angle_difference)
            add_point(angle)
            add_point(angle + math.pi)
    
    # generate events
    # event 1: getting closer
    # event 2: getting further
    for e in everything:
        if d1(e[0], positions[i][j][0]) > 1e-6:
            angle = math.atan2(e[1][1], e[1][0])
            r1 = math.hypot(positions[i][j][1][0], positions[i][j][1][1])
            r2 = math.hypot(e[1][0], e[1][1])
            add_event(angle, {"point": e[1], "event": "getting further", "multiplier": min(r1,r2) / d1(e[0], positions[i][j][0])})
            add_event(angle + math.pi, {"point": e[1], "event": "getting closer", "multiplier": min(r1,r2) / d1(e[0], positions[i][j][0])})
    
    # event 3: start forbidden (open interval)
    # event 4: stop forbidden (open interval)
    for e in everything:
        if d1(e[0], positions[i][j][0]) > 1e-6:
            angle = math.atan2(e[1][1], e[1][0])
            angle_difference = min_angle_diff(positions[i][j], e)
            angle_difference = min(angle_difference, math.pi)
            if angle_difference > 0:
                add_event(angle - angle_difference, {"point": e[1], "event": "start forbidden"})
                add_event(angle + angle_difference, {"point": e[1], "event": "stop forbidden"}) 
    
    k = len(positions[i])
    if k>=3:
        # event 5: start allowing placement (open interval)
        angle1 = math.atan2(positions[i][(j-1+k)%k][1][1], positions[i][(j-1+k)%k][1][0])
        add_event(angle1, {"point": positions[i][(j-1+k)%k][1], "event": "start allowing placement"})
        # event 6: stop allowing placement (open interval)
        angle2 = math.atan2(positions[i][(j+1)%k][1][1], positions[i][(j+1)%k][1][0])
        add_event(angle2, {"point": positions[i][(j+1)%k][1], "event": "stop allowing placement"})
        assert(abs(angle1-angle2)%(2*math.pi) > 1e-6)


# O(nlogn)
def initialize_adjust(i, j, positions, d1, d2):
    change_in_distortion = 0
    number_of_forbidden = 0
    placement_allowed = False
    if (len(positions[i]) <= 2):
        placement_allowed = True
    forbidden_list = {}

    for k in sorted(critical_points.keys()):
        for event in critical_points[k]:
            if event["event"] == "start allowing placement":
                placement_allowed = True
            if event["event"] == "stop allowing placement":
                placement_allowed = False
            if event["event"] == "start forbidden":
                number_of_forbidden += 1
                forbidden_list[event["point"]] = True
            if event["event"] == "stop forbidden":
                if event["point"] in forbidden_list:
                    del forbidden_list[event["point"]]
                    number_of_forbidden -= 1
            if event["event"] == "getting further":
                if k<math.pi:
                    change_in_distortion -= event["multiplier"]
                else:
                    change_in_distortion += event["multiplier"]
    
    global loop_iteration
    #plotting.create_debug_photo(positions, i, j, critical_points, loop_iteration, test_name, False)
    
    return change_in_distortion, number_of_forbidden, placement_allowed

# O(nlogn)
def adjust(i, j, positions, d1, d2):
    generate_critical_points(i, j, positions, d1, d2)
    change_in_distortion, number_of_forbidden, placement_allowed = initialize_adjust(i, j, positions, d1, d2)
    init_change_in_distortion, init_number_of_forbidden, init_placement_allowed = change_in_distortion, number_of_forbidden, placement_allowed
    min_adjust = 4e18
    current_adjust = 0
    old_angle = math.atan2(positions[i][j][1][1], positions[i][j][1][0])
    best_angle = old_angle
    
    list_of_keys = sorted(critical_points.keys())
    
    for it in range(len(list_of_keys)):
        k = list_of_keys[it]


        for event in critical_points[k]:
            if event["event"] == "stop forbidden":
                number_of_forbidden -= 1
            if event["event"] == "stop allowing placement":
                placement_allowed = False
        
        if current_adjust < min_adjust and (placement_allowed and number_of_forbidden == 0) or (same_angle(k, old_angle)):
            min_adjust = current_adjust
            best_angle = k
        
        for event in critical_points[k]:
            if event["event"] == "start allowing placement":
                placement_allowed = True
            if event["event"] == "start forbidden":
                number_of_forbidden += 1
            if event["event"] == "getting further":
                change_in_distortion += 2*event["multiplier"]
            if event["event"] == "getting closer":
                change_in_distortion -= 2*event["multiplier"]
        
        if it+1 < len(list_of_keys):
            next_key = list_of_keys[it+1]
        else:
            next_key = 2*math.pi
        
        current_adjust += change_in_distortion*(next_key-k)
    
    if abs(current_adjust)>=1e-2:
        print("current_adjust = {}".format(current_adjust))
        pprint.pprint(critical_points)
    assert(abs(current_adjust) < 1e-2)

    r = math.hypot(positions[i][j][1][0], positions[i][j][1][1])
    positions[i][j] = (positions[i][j][0], (r*math.cos(best_angle), r*math.sin(best_angle)))

    #tmp = is_non_contracting(positions, d1, d2)
    #global test_name, loop_iteration
    #if not tmp:
    #    print(test_name, loop_iteration, i, j, old_angle, best_angle)
    #    print(positions[i][j])
    #assert(tmp)

    #plotting.create_debug_photo(positions, i, j, critical_points, loop_iteration, test_name, True)
    return 1 if (abs(old_angle%(2*math.pi)-best_angle%(2*math.pi))%(2*math.pi)>1e-6) else 0

# O(C(2*n^2 + n^2log(n)))
def optimize_placement(positions, d1, d2):
    global adjustments, loop_iteration, test_name
    loop_iteration = 0
    adjustments = 0
    distortion = 0
    n = 0
    while True:
        loop_iteration += 1
        print("Start of loop_iteration {}".format(loop_iteration))
        start_time = timer()
        
        everything = list(itertools.chain.from_iterable(positions))
        
        n = len(everything)
        k = len(positions)
        change_in_distortion = 0
        distortion = total_distortion(positions, d1, d2)

        for l in range(100):
            i = random.randint(0,k-1)
            if len(positions[i]) == 0:
                continue
            j = random.randint(0,len(positions[i])-1)
            tmp = adjust(i,j,positions,d1,d2)
            change_in_distortion += tmp
        
        new_distortion = total_distortion(positions, d1, d2)
        adjustments += change_in_distortion
        end_time = timer()
        print("End of loop_iteration {}. Elapsed time = {}".format(loop_iteration, end_time-start_time))
        if change_in_distortion == 0 or new_distortion/distortion>=0.99 or loop_iteration >= 20:
            break
    
    distortion = total_distortion(positions, d1, d2)
    
    print("Optimize placement runs {} times".format(loop_iteration))
    print("Number of adjustments: {}".format(adjustments))
    return (distortion/(n*(n-1)) *2 , positions)

# O(n)
def correct_positions(final_result, radius):
    def correct(position, r):
        l = math.hypot(position[0], position[1])
        position = (position[0] / l * r, position[1] / l * r)
        return position
    
    for i in range(len(radius)):
        for j in range(len(final_result[i])):
            final_result[i][j] = (final_result[i][j][0], correct(final_result[i][j][1], 0.1*(2 ** (i+1))))
    return final_result

def fit_transform(distance_matrix, min_dist=0, number_of_rings=4, with_optimizer=True):
    focus = 0
    entities = list(range(1, len(distance_matrix)))
    d1 = lambda x,y: max(0.1, distance_matrix[x][y]) if x!=y else 0
    radius = find_radii(focus, entities, d1, number_of_rings)

    groups = partition(focus, entities, radius, d1) #O(nlog(g))
    for i in range(len(groups)):
        print("Group {}: {}".format(i, len(groups[i])))
        groups[i] = embed_to_circle(groups[i],d1) # O(g*2^(n/g)*(n/g))
    
    current_positions = rotational_align(groups, radius, d1, d2) # O(20n^2)
    
    print("Finish rotational alignment")
    current_positions = ensure_non_contracting(current_positions, d1, d2) # O(n^2)
    print("Finish ensure contracting")
    assert(is_non_contracting(current_positions, d1, d2)) # O(n^2)
    final_result = current_positions
    if with_optimizer:
        print("Start optimize placement")
        final_result = optimize_placement(current_positions, d1, d2)[1] # O(C(2*n^2 + n^2log(n))
        print("Finish optimize placement")
    final_result = correct_positions(final_result, radius) #O(n)
    everything = list(itertools.chain.from_iterable(final_result)) #O(n)
    everything.sort(key=lambda x: x[0])
    everything = [[0,0]] + list(map(lambda x: [x[1][0], x[1][1]], everything))
    return ensure_min_dist(everything, min_dist) #O(n^2)


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
















