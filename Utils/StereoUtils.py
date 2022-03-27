import cv2

def splitMergedImage(image):

    if image is None:
        return None, None

    height = image.shape[0]
    width = image.shape[1]

    width_cutoff = width // 2

    s1 = image[:, :width_cutoff]
    s2 = image[:, width_cutoff:]

    return s1, s2