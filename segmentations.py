import numpy as np
import cv2 as cv
from utils import ROWS, COLS, CHANNELS, RGB_CHANNELS, get_pixel_neighborhood_indices

def simple_gradient(img: cv.Mat):

    if img.shape[CHANNELS] == RGB_CHANNELS:
        img = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
        print("Converted to grayscale")

    rows = img.shape[ROWS]
    cols = img.shape[COLS]

    Gx = np.zeros((rows, cols), dtype=np.float32)
    Gy = np.zeros((rows, cols), dtype=np.float32)

    mask_X = np.array([-1, 1])
    mask_Y = np.array([[-1], [1]])

    result = np.zeros((rows, cols), dtype=np.float32)

    for i in range(rows):
        for j in range(cols):
            if j == cols - 1:
                Gx[i, j] = img[i, j] * mask_X[0][0]
            else:
                Gx[i, j] = img[i, j] * mask_X[0][0] + img[i, j + 1] * mask_X[0][1]

            if i == rows - 1:
                Gy[i, j] = img[i, j] * mask_Y[0][0]
            else:
                Gy[i, j] = img[i, j] * mask_Y[0][0] + img[i + 1, j] * mask_Y[1][0]

            result[i, j] = max(abs(Gx[i, j]), abs(Gy[i, j]))

    return [result, Gx, Gy]

def laplacien_connenctivity(img: cv.Mat, connectivity, threshold) :
    if img.shape[CHANNELS] == RGB_CHANNELS:
        img = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
        print("Converted to grayscale")

    rows = img.shape[ROWS]
    cols = img.shape[COLS]

    result = np.zeros((rows, cols), dtype=np.float32)

    if connectivity == 4:
        mask = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
    elif connectivity == 8:
        mask = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]])
    else:
        print("Connectivity must be 4 or 8")
        return

    for i in range(rows):
        for j in range(cols):
            if i == 0 or j == 0 or i == rows - 1 or j == cols - 1:
                result[i, j] = 0
            else:
                result[i, j] =  img[i - 1, j - 1] * mask[0][0] + img[i - 1, j] * mask[0][1] + img[i - 1, j + 1] * mask[0][2] + \
                                img[i, j - 1] * mask[1][0] + img[i, j] * mask[1][1] + img[i, j + 1] * mask[1][2] + \
                                img[i + 1, j - 1] * mask[2][0] + img[i + 1, j] * mask[2][1] + img[i + 1, j + 1] * mask[2][2]

    result = np.where(result > threshold, 255, 0)

    return result


def locals_maximals(img: cv.Mat):
    if img.shape[CHANNELS] == RGB_CHANNELS:
        img = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
        print("Converted to grayscale")

    [norm, Gx, Gy] = simple_gradient(img)

    rows = img.shape[ROWS]
    cols = img.shape[COLS]

    # convert all 0 in Gx to 1 to avoid division by 0
    Gx = np.where(Gx == 0, 1, Gx)
    orientation = np.arctan(Gy / Gx)
    
    # convert orientation to degrees
    orientation = np.rad2deg(orientation)

    # convert orientation to 0, 45, 90, 135
    orientation = np.where(orientation < 0, orientation + 180, orientation)

    result = np.zeros((rows, cols), dtype=np.float32)

    for i in range(rows):
        for j in range(cols):
            angle = orientation[i, j]

            index1 = [1,1]
            index2 = [1,1]

            if (angle >= 0 and angle < 22) or (angle > 157 and  angle <= 202) or angle > 337:
                index1 = [i, j - 1]
                index2 = [i, j + 1]
            elif (angle >= 22 and angle <= 66) or (angle > 202 and angle <= 246):
                index1 = [i - 1, j -1]
                index2 = [i + 1, j - 1]
            elif (angle > 66 and angle <= 112) or (angle > 246 and angle <= 292):
                index1 = [i - 1, j]
                index2 = [i + 1, j]
            elif (angle > 112 and angle <= 157) or (angle > 292 and angle <= 337):
                index1 = [i - 1, j - 1]
                index2 = [i + 1, j + 1]
            else :
                print("Error in orientation")

            if norm[i, j] > norm[index1[0], index1[1]] and norm[i, j] > norm[index2[0], index2[1]]:
                result[i, j] = norm[i, j]
            
    return [result, norm]


def hysteresys(img: cv.Mat, threshold1, threshold2):
    if img.shape[CHANNELS] == RGB_CHANNELS:
        img = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
        print("Converted to grayscale")

    rows = img.shape[ROWS]
    cols = img.shape[COLS]

    [max_locals, norm] = locals_maximals(img)

    result = np.zeros((rows, cols), dtype=np.float32)

    stack = []

    for i in range(rows):
        for j in range(cols):
            if max_locals[i, j] >= threshold1:
                result[i, j] = 255
                stack.append([i, j])
    

    while len(stack) > 0:
        [i, j] = stack.pop()

        neighborhood = get_pixel_neighborhood_indices(img, i, j, 8)

        for k in range(len(neighborhood)):
            [n_i, n_j] = neighborhood[k]

            if norm[n_i, n_j] >= threshold2 and result[n_i, n_j] == 0:
                result[n_i, n_j] = 255
                stack.append([n_i, n_j])

    return result
