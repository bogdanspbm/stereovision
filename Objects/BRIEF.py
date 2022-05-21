import cv2
import random
from Objects.KeyPoint import KeyPoint, getHausdorfDistance

'''
This class implements default BRIEF detection
The constructor receives next parameters:
---------------------------------------------
radius - half size of scanning window
pairs_count - the amount of compared pairs
'''


class DetectorBRIEF():

    def __init__(self, radius=15, pairs_count=256):
        self.pairs = []
        self.radius = radius
        self.pairs_count = pairs_count
        for i in range(pairs_count):
            a_x, a_y = random.randint(-radius, radius), random.randint(-radius, radius)
            b_x, b_y = random.randint(-radius, radius), random.randint(-radius, radius)
            self.pairs.append([[a_x, a_y], [b_x, b_y]])

    '''
    This method receives an image and points array
    and returns an array of points features
    '''

    def getParamVectors(self, image, points):
        # Apply blur for flatting result
        blur = cv2.GaussianBlur(image, (5, 5), 0)
        vectors = []

        # Generate an image with black borders for simplify calculation
        process_image = cv2.copyMakeBorder(
            blur,
            self.radius,
            self.radius,
            self.radius,
            self.radius,
            cv2.BORDER_CONSTANT,
            value=(0, 0, 0))

        # Calculate features
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

    '''
    This method receives two arrays and looks for matches between pairs
    You can also change a limit value for understanding that points are similar
    '''

    def calculatePairs(self, key_points_a, key_points_b, limit=25):

        pairs = []

        key_points_c = key_points_b.copy()

        for keypoint_a in key_points_a:
            min_distance = -1
            pair_point = None
            for keypoint_b in key_points_c:
                distance = getHausdorfDistance(keypoint_a, keypoint_b)
                if min_distance == -1 or min_distance > distance:
                    min_distance = distance
                    pair_point = keypoint_b
            if min_distance <= limit:
                pairs.append([keypoint_a, pair_point])
                key_points_c.remove(pair_point)

        return pairs

