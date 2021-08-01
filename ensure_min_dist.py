import math
def ensure_min_dist(result, min_dist):
    rings = {}
    for p in result[1:]:
        r = math.hypot(p[0], p[1])
        flag = False
        for x in rings:
            if abs(r-x) < 1e-6:
                rings[x].append(p)
                flag = True
        if not flag:
            rings[r] = [p]
    
    max_r = max(list(rings.keys()))

    def angle_dist(x,y):
        angle1 = math.atan2(x[1], x[0])
        angle2 = math.atan2(y[1], y[0])
        return (angle2 - angle1) % (2*math.pi)
    
    for R in rings:
        r = R / max_r
        min_angle = math.acos((r ** 2 + r**2 - min_dist**2) / (2*(r**2)))
        rings[R].sort(key=lambda x: math.atan2(x[1], x[0]))
        parts = []
        total = math.pi*2
        for i in range(len(rings[R])):
            if angle_dist(rings[R][i-1], rings[R][i]) <= min_angle:
                total -= min_angle
            else:
                parts.append(angle_dist(rings[R][i-1], rings[R][i]))
            
        total_parts = sum(parts)

        print("total parts = {}".format(total_parts))
        print("total = {}".format(total))
        current_angle = math.atan2(rings[R][0][1], rings[R][0][0])
        result[result.index(rings[R][0])] = [r*math.cos(current_angle), r*math.sin(current_angle)]
        for t in range(len(rings[R])-1):
            angle = angle_dist(rings[R][t], rings[R][t+1])
            if angle <= min_angle:
                current_angle += min_angle
            else:
                current_angle += angle / total_parts * total
            i = t+1
            j = result.index(rings[R][i])
            result[j] = [r*math.cos(current_angle), r*math.sin(current_angle)]
    
    return result


        