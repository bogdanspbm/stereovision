from Objects.KeyPoint import getHausdorfDistance, KeyPoint


class HorizontalDetector():
    def __init__(self, height=512, mode="DEFAULT"):
        self.height = height
        self.mode = mode
        pass

    def getParamVectors(self, image, points):
        vectors = []
        if self.mode == "DEFAULT":
            for point in points:
                vec = []
                for i in range(-self.height, self.height):
                    vec.append(image[point[0]][point[1] + i])
                vectors.append(KeyPoint(point, vec))

        return vectors

    def calculatePairs(self, keypoints_a, keypoints_b):

        pairs = []

        keypoints_c = keypoints_b.copy()

        for keypoint_a in keypoints_a:
            min_distance = -1
            pair_point = None
            for keypoint_b in keypoints_c:
                distance = getHausdorfDistance(keypoint_a, keypoint_b)
                if min_distance == -1 or min_distance > distance:
                    min_distance = distance
                    pair_point = keypoint_b
            pairs.append([keypoint_a, pair_point])
            keypoints_c.remove(pair_point)

        return pairs
