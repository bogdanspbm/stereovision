import math
import cv2
import numpy as np
from math import sqrt


def findExtremums(vec, radius=5, separate=True, merge=True, merge_limit=2, limit=0):
    minimums = []
    maximums = []
    extremums = []

    for i in range(len(vec)):
        flag_minimum = True
        flag_maximum = True
        el = vec[i]
        for k in range(-radius, radius):
            if k is not 0:
                if i + k >= 0 and i + k < len(vec):
                    sec_el = vec[i + k]
                    if el > sec_el:
                        flag_minimum = False
                    if el < sec_el:
                        flag_maximum = False

                    if flag_minimum == False and flag_maximum == False:
                        break
        if flag_minimum and abs(el) >= limit:
            if separate:
                minimums.append([i, el])
            else:
                extremums.append((i, el, "min"))
        if flag_maximum and abs(el) >= limit:
            if separate:
                maximums.append([i, el])
            else:
                extremums.append((i, el, "max"))

    if merge:
        delete_index = []
        if separate:
            for i in range(len(minimums) - 1):
                a = minimums[i]
                b = minimums[i + 1]
                if abs(a[0] - b[0]) <= radius and (abs(a[1] - b[1]) < merge_limit):
                    delete_index.append(i + 1)

            N = len(delete_index)
            for i in range(N):
                minimums.pop(delete_index[N - i - 1])

            delete_index = []

            for i in range(len(maximums) - 1):
                a = maximums[i]
                b = maximums[i + 1]
                if abs(a[0] - b[0]) <= radius and (abs(a[1] - b[1]) < merge_limit):
                    delete_index.append(i + 1)

            N = len(delete_index)
            for i in range(N):
                maximums.pop(delete_index[N - i - 1])
        else:
            for i in range(len(extremums) - 1):
                a = extremums[i]
                b = extremums[i + 1]
                if abs(a[0] - b[0]) <= radius and (abs(a[1] - b[1]) < merge_limit):
                    delete_index.append(i + 1)

            N = len(delete_index)
            for i in range(N):
                extremums.pop(delete_index[N - i - 1])

    if separate:
        return minimums, maximums
    else:
        return extremums


def findDisparity(vec_a, vec_b):
    N = len(vec_a)
    pairs = []
    min_sum_dist = -1
    best_disp = 0
    for i in range(N):
        disp = vec_a[i][0]
        tmp_sum_dist = 0
        for a in vec_a:
            min_dist = -1
            el = None
            for b in vec_b:
                dist = ((a[0] - disp - b[0]) ** 2 + (a[1] - b[1]) ** 2)
                if min_dist == -1 or dist < min_dist:
                    min_dist = dist
                    el = b
            pairs.append(el)
            tmp_sum_dist += min_dist

        if min_sum_dist == -1 or tmp_sum_dist < min_sum_dist:
            min_sum_dist = tmp_sum_dist
            best_disp = disp

    return best_disp, min_sum_dist


def findPairs(vec_a, vec_b, disp):
    pairs = []
    min_sum_dist = -1
    best_disp = 0

    res = []
    edges_a = []
    edges_b = []

    for i in range(len(vec_a) - 1):
        a_1 = vec_a[i].copy()
        a_2 = vec_a[i + 1].copy()
        edges_a.append([(a_1[0] + a_2[0]) / 2, (a_1[1] + a_2[1]) / 2])

    for i in range(len(vec_b) - 1):
        b_1 = vec_b[i].copy()
        b_2 = vec_b[i + 1].copy()
        edges_b.append([(b_1[0] + b_2[0]) / 2, (b_1[1] + b_2[1]) / 2])

    return res


def getDistanceBetweenEdges(edge_a, edge_b, disp, a=1, b=1, c=1):
    if edge_a[0][2] != edge_b[0][2] or edge_a[1][2] != edge_b[1][2]:
        return -1

    width_a = abs(edge_a[1][0] - edge_a[0][0])
    width_b = abs(edge_b[1][0] - edge_b[0][0])

    height_a = edge_a[1][1] - edge_a[0][1]
    height_b = edge_b[1][1] - edge_b[0][1]

    E_a = (edge_a[0][0] - disp - edge_b[0][0]) ** 2 + (edge_a[0][1] - edge_b[0][1]) ** 2 + (
            edge_a[1][0] - disp - edge_b[1][0]) ** 2 + (edge_a[1][1] - edge_b[1][1]) ** 2
    E_b = (width_a - width_b) ** 2
    E_c = (height_a - height_b) ** 2

    return a * E_a + b * E_b + c * E_c


def matchExtremus(image_a, image_b, extr_a, extr_b, radius=20, dist_limit=1000):
    pairs = []

    image_a_copy = cv2.copyMakeBorder(
        image_a,
        radius,
        radius,
        radius,
        radius,
        cv2.BORDER_CONSTANT,
        value=(0, 0, 0))

    image_b_copy = cv2.copyMakeBorder(
        image_b,
        radius,
        radius,
        radius,
        radius,
        cv2.BORDER_CONSTANT,
        value=(0, 0, 0))

    for a in extr_a:
        min_dist = -1
        min = None
        for b in extr_b:
            if a[2] == b[2] and a[0] > b[0]:
                block_a = image_a_copy[a[1]:a[1] + 2 * radius, a[0]:a[0] + 2 * radius]
                block_b = image_b_copy[b[1]:b[1] + 2 * radius, b[0]:b[0] + 2 * radius]
                dist = getMatrixDistance(block_a, block_b)
                if min_dist == -1 or dist < min_dist:
                    min_dist = dist
                    min = b

        if min is not None and min_dist < dist_limit:
            pairs.append([a, min])

    return pairs


def getVecDistance(vec_a, vec_b):
    sum = 0
    for i in range(len(vec_a)):
        a = vec_a[i]
        b = vec_b[i]
        sum += (a - b) ** 2

    return math.sqrt(sum)


def getMatrixDistance(mat_a, mat_b):
    return math.sqrt(np.sum(np.absolute(mat_a - mat_b) ** 2))


def generateExtremumPoints(extremums, line, height):
    points = []

    for ext in extremums:
        x = ext[0]
        y = (int)((-line[0] * x - line[2]) / (line[1]))
        if y >= 0 and y < height:
            points.append([x, y, ext[2]])

    return points


def getVectorNorm(vec):
    norm = sqrt(vec[0] ** 2 + vec[1] ** 2 + vec[2] ** 2)
    return norm
