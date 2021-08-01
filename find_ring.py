def closest(x, radius):
    for i in range(1, len(radius)):
        if (x>radius[i-1] and x<=radius[i]):
            return i
    return 0
