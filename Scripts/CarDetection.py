import cv2
import numpy as np
from Utils.MathUtils import getMatrixDistance, getVectorNorm
from Utils.FileUtils import loadProjectionMatrix
from Utils.StereoUtils import getWorldCoordinates

'''
This is a test script for car detection and calculating distances to cars
Download your own video into 'Videos' folder to test it
'''


def start():
    car_cascade = cv2.CascadeClassifier('../Cascades/cars.xml')
    vid_capture = cv2.VideoCapture('../Videos/video_1.mov')
    object_detector = cv2.createBackgroundSubtractorMOG2()

    WIDTH = 3840
    HEIGHT = 1080

    P_1, P_2 = loadProjectionMatrix()

    while True:
        ret, frame = vid_capture.read()
        # frame = cv2.GaussianBlur(frame, (5, 5), 0)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        # Detect cars
        cars = car_cascade.detectMultiScale(gray, 1.1, 1)
        cars = clearHeight(cars, HEIGHT)
        cars = mergeCars(cars)

        ncars = 0

        # Detect lines

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        low_yellow = np.array([0, 0, 150])
        up_yellow = np.array([50, 50, 255])
        mask = cv2.inRange(hsv, low_yellow, up_yellow)
        edges = cv2.Canny(mask, 75, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 50, maxLineGap=50)
        lines = filterLines(lines, WIDTH)
        if lines is not None and len(lines) > 0:
            for line in lines:
                if line is not None:
                    x1, y1, x2, y2 = line[0]
                    cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 5)

        cars = getCanrsInInterestZone(cars, lines, WIDTH)

        cars_left, cars_right = splitCars(cars, WIDTH)

        blocks_left = []
        blocks_right = []

        # Draw border

        block_size = 20

        for (x, y, w, h) in cars_left:
            x_c = (int)(x + w / 2)
            y_c = (int)(y + h / 2)

            block = gray[y_c - block_size:y_c + block_size, x_c - block_size:x_c + block_size]
            blocks_left.append(block)

        for (x, y, w, h) in cars_right:
            x_c = (int)(x + w / 2)
            y_c = (int)(y + h / 2)

            block = gray[y_c - block_size:y_c + block_size, x_c - block_size:x_c + block_size]
            blocks_right.append(block)

        pairs = compareBlocks(blocks_left, blocks_right)

        for (i, k) in pairs:
            if k != -1 and k is not None:
                (x, y, w, h) = cars_right[k]
                point_right = np.array([x + w / 2 - WIDTH / 2, y + h / 2])
                (x, y, w, h) = cars_left[i]
                point_left = np.array([x + w / 2, y + h / 2])

                dist = (int)(getVectorNorm(getWorldCoordinates(P_1, P_2, point_left, point_right)))

                cv2.circle(gray, ((int)(x + w / 2), (int)(y + h / 2)), 10, (0, 0, 255), 2)
                cv2.rectangle(gray, (x, y), (x + w, y + h), (0, 0, 255), 2)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(gray, 'Dist: ' + str(dist), (x + 6, y - 6), font, 0.5, (0, 0, 255), 1)

                (x, y, w, h) = cars_right[k]
                cv2.rectangle(gray, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.circle(gray, ((int)(x + w / 2), (int)(y + h / 2)), 10, (0, 0, 255), 2)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(gray, 'Dist: ' + str(dist), (x + 6, y - 6), font, 0.5, (0, 0, 255), 1)

                ncars = ncars + 1

        # Show image
        resized = cv2.resize(gray, (1920, 540), interpolation=cv2.INTER_AREA)
        cv2.imshow("Car Detection", resized)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break


def compareBlocks(blocks_left, blocks_right):
    pairs = []
    for i in range(len(blocks_left)):
        a = blocks_left[i]
        c = None
        min_dist = -1
        for k in range(len(blocks_right)):
            b = blocks_right[k]
            try:
                dist = getMatrixDistance(a, b)
            except:
                dist = dist
            if min_dist == -1 or dist < min_dist:
                min_dist = dist
                c = k
        pairs.append((i, c))
    return pairs


# Returns True if point is lefter than line
# else returns False
def getPointRelaiveLine(point, line):
    if line is not None:
        x1, y1, x2, y2 = line[0]
        x, y = point

        x_line = (y - y1) * (x2 - x1) / (y2 - y1) + x1

        return x_line > x

    return True


def getCanrsInInterestZone(arr, lines, width):
    res = []
    for (x, y, w, h) in arr:
        flag = False
        if x < width / 2:
            flag_a = getPointRelaiveLine((x + w / 2, y + h / 2), lines[1])
            flag_b = getPointRelaiveLine((x + w / 2, y + h / 2), lines[0])

        else:
            flag_a = getPointRelaiveLine((x + w / 2, y + h / 2), lines[3])
            flag_b = getPointRelaiveLine((x + w / 2, y + h / 2), lines[2])

        if flag_a == True and flag_b == False:
            flag = True

        if flag:
            res.append((x, y, w, h))

    return res


def filterLines(lines, width):
    image_split = (int)(width / 2)
    left_center = (int)(image_split / 2)
    right_center = image_split + left_center

    a_left_line = None
    a_right_line = None
    b_left_line = None
    b_right_line = None

    for line in lines:
        x1, y1, x2, y2 = line[0]
        length = (x2 - x1) ** 2 + (y2 - y1) ** 2
        if length > 80000:
            line_center = abs(x1 + x2) / 2
            if line_center < image_split:
                if line_center < left_center:
                    if a_left_line is None:
                        a_left_line = line
                    else:
                        xa, _, xb, _ = a_left_line[0]
                        if left_center - line_center < left_center - abs(xb - xa) / 2:
                            a_left_line = line
                else:
                    if a_right_line is None:
                        a_right_line = line
                    else:
                        xa, _, xb, _ = a_right_line[0]
                        if line_center - left_center < abs(xb - xa) / 2 - left_center:
                            a_right_line = line
            else:
                if line_center < right_center:
                    if b_left_line is None:
                        b_left_line = line
                    else:
                        xa, _, xb, _ = b_left_line[0]
                        if right_center - line_center < right_center - abs(xb - xa) / 2:
                            b_left_line = line
                else:
                    if b_right_line is None:
                        b_right_line = line
                    else:
                        xa, _, xb, _ = b_right_line[0]
                        if line_center - right_center < abs(xb - xa) / 2 - right_center:
                            b_right_line = line
                        pass

            # result.append(line)

    result = [a_left_line, a_right_line, b_left_line, b_right_line]

    return result


def splitCars(arr, width):
    arr_left = []
    arr_right = []

    for (x, y, w, h) in arr:
        if x < width / 2:
            arr_left.append((x, y, w, h))
        else:
            arr_right.append((x, y, w, h))

    return arr_left, arr_right


def clearHeight(arr, height):
    res = []
    for (x, y, w, h) in arr:
        if y + h / 2 > height / 2:
            res.append((x, y, w, h))
    return res


def mergeCars(arr):
    res = arr.copy()
    merged = []
    for a in arr:
        (a_x, a_y, a_w, a_h) = a
        if a not in merged:
            for b in arr:
                if b not in merged:
                    (b_x, b_y, b_w, b_h) = b
                    c_x = b_x + b_w / 2
                    c_y = b_y + b_h / 2
                    if c_x > a_x and c_x < a_x + a_w:
                        if c_y > a_y and c_y < a_y + a_h:
                            if b_w * b_h < a_w * a_h:
                                merged.append(b)

    for el in merged:
        res.remove(el)

    return res


def clearDuplicates(arr):
    i = 0
    N = len(arr)
    while i < N:
        a = arr[i]
        pairs = []
        (x_a, y_a, w_a, h_a) = a
        for b in arr:
            (x_b, y_b, w_b, h_b) = b
            dist = (x_a + w_a - x_b - w_b) ** 2 + (y_a + h_a - y_b - h_b) ** 2
            if dist < 100000:
                pairs.append(b)

        if len(pairs) > 3:
            for el in pairs:
                arr.remove(el)

        N = len(arr)
        i += 1

    return arr


start()
