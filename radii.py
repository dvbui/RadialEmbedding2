import math
def find_radii(focus, entities, d1, number_of_rings):
    radii = [0.1*(2**i) for i in range(number_of_rings)]
    total = sum(radii)
    n = len(entities)
    point_per_rings = []
    for i in range(number_of_rings):
        point_per_rings.append(int(math.floor(n*radii[i] /total)+0.5))
    remainder = int(math.floor(n - sum(point_per_rings)) + 0.5)
    for i in range(remainder):
        point_per_rings[i%number_of_rings] += 1

    entities.sort(key=lambda x: d1(focus, x))
    radius_result = []
    prev = 0
    for i in range(number_of_rings):
        prev += point_per_rings[i]
        radius_result.append(d1(focus, entities[prev-1]))
    return radius_result