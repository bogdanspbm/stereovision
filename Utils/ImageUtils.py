import cv2

'''
This method splits an input image on two by vertical 
and returns them
'''


def splitMergedImage(image):
    if image is None:
        return None, None

    width = image.shape[1]

    width_cutoff = width // 2

    s1 = image[:, :width_cutoff]
    s2 = image[:, width_cutoff:]

    return s1, s2


'''
This method receives an image and line params
and returns an image with drawn line 
'''


def drawEpiline(image, line):
    height, width = image.shape
    a = line[0][0][0]
    b = line[0][0][1]
    c = line[0][0][2]
    x0, y0 = map(int, [0, -c / b])
    x1, y1 = map(int, [width, -(c + a * width) / b])
    print((x0, y0), (x1, y1))
    cv2.line(image, (x0, y0), (x1, y1), (0, 255, 0), thickness=1)
    return image


'''
This method receives an image and distance value
and returns an image with drawn distance 
'''


def imageDrawDistance(image, distance):
    value = "Distance: " + str((int)(distance)) + " cm"
    cv2.putText(image, value, (25, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255))


'''
This method receives an image and line start/end points
and returns all pixels between input points
'''


def getScanline(image, point_start, point_end):
    height, width = image.shape
    scanline = []

    a_x = point_start[0]
    a_y = point_start[1]

    b_x = point_end[0]
    b_y = point_end[1]

    if a_y == b_y:
        if a_y < height:
            return image[a_y, :]
        else:
            return []

    if a_y >= height and b_y >= height:
        return []

    for x in range(a_x, b_x):
        y = (int)((x - a_x) / (b_x - a_x) * (b_y - a_y) + a_y)
        if y < height and y >= 0:
            scanline.append(image[y, x])

    return scanline
