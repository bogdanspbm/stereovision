class KeyPoint():
    def __init__(self, coord, param):
        self.coord = coord
        self.param = param

    def getCoord(self):
        return (self.coord[0], self.coord[1])

    def getParam(self):
        return self.param


def getHausdorfDistance(keypoin_a, keypoint_b):
    vec_a = keypoin_a.getParam()
    vec_b = keypoint_b.getParam()
    distance = 0
    for i in range(len(vec_a)):
        if vec_a[i] != vec_b[i]:
            distance += 1
    return distance
