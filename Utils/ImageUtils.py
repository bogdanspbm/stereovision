import cv2


def removeImageBorder(image, border=0):
    if image is None:
        return None

    height = image.shape[0]
    width = image.shape[1]

    s1 = image[:, :width - border]
    s2 = s1[:, border:]

    return s2


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
