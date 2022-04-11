import cv2
import random
from Objects.KeyPoint import KeyPoint, getHausdorfDistance
from Objects.Timer import Timer


class DetectorBRIEF():

    def __init__(self, radius=15, pairs_count=256):
        self.pairs = []
        self.radius = radius
        self.pairs_count = pairs_count
        for i in range(pairs_count):
            a_x, a_y = random.randint(-radius, radius), random.randint(-radius, radius)
            b_x, b_y = random.randint(-radius, radius), random.randint(-radius, radius)
            self.pairs.append([[a_x, a_y], [b_x, b_y]])

    def getParamVectors(self, image, points):
        blur = cv2.GaussianBlur(image, (5, 5), 0)
        vectors = []

        process_image = cv2.copyMakeBorder(
            blur,
            self.radius,
            self.radius,
            self.radius,
            self.radius,
            cv2.BORDER_CONSTANT,
            value=(0, 0, 0))

        for point in points:
            vec = []
            for pair in self.pairs:
                a_x = point[0] + pair[0][0]
                a_y = point[1] + pair[0][1]
                b_x = point[0] + pair[1][0]
                b_y = point[1] + pair[1][1]

                pixel_a = process_image[a_y][a_x]
                pixel_b = process_image[b_y][b_x]

                if pixel_a > pixel_b:
                    vec.append(1)
                else:
                    vec.append(0)

            kp = KeyPoint(point, vec)
            vectors.append(kp)

        return vectors

    def calculatePairs(self, keypoints_a, keypoints_b, border=25):

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
            if min_distance <= border:
                pairs.append([keypoint_a, pair_point])
                keypoints_c.remove(pair_point)

        return pairs


class DetectorHorizontalBRIEF():
    def __init__(self, width=1, height=128, pairs_count=128):
        self.pairs = []
        self.width = width
        self.height = height
        self.pairs_count = pairs_count

        for i in range(pairs_count):
            a_x, a_y = random.randint(-width, width), random.randint(-height, height)
            b_x, b_y = random.randint(-width, width), random.randint(-height, height)
            self.pairs.append([[a_x, a_y], [b_x, b_y]])

    def getParamVectors(self, image, points):
        blur = cv2.GaussianBlur(image, (5, 5), 0)
        vectors = []

        process_image = cv2.copyMakeBorder(
            blur,
            self.height,
            self.height,
            self.width,
            self.width,
            cv2.BORDER_CONSTANT,
            value=(0, 0, 0))

        for point in points:
            vec = []
            for pair in self.pairs:
                a_x = point[0] + pair[0][0]
                a_y = point[1] + pair[0][1]
                b_x = point[0] + pair[1][0]
                b_y = point[1] + pair[1][1]

                pixel_a = process_image[a_y][a_x]
                pixel_b = process_image[b_y][b_x]

                if pixel_a > pixel_b:
                    vec.append(1)
                else:
                    vec.append(0)

            kp = KeyPoint(point, vec)
            vectors.append(kp)

        return vectors

    def calculatePairs(self, keypoints_a, keypoints_b, border=25):

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
            if min_distance <= border:
                pairs.append([keypoint_a, pair_point])
                keypoints_c.remove(pair_point)

        return pairs
