import numpy as np
import cv2 as cv
from utils import ROWS, COLS, RGB_CHANNELS

def dilate(img: cv.Mat) :
    
    if(len(img.shape) == RGB_CHANNELS) :
        return Exception("Img need to be grayscale and binarzed")

    rows = img.shape[ROWS]
    cols = img.shape[COLS]

    result = img.copy()

    for i in range(1, rows - 1) :
        for j in range(1, cols - 1) :
            pixel = img[i,j]
            if not pixel == 255 : continue

            result[i + 1, j] = 255
            result[i, j + 1] = 255
            result[i, j - 1] = 255
            result[i - 1, j] = 255

    return result     
            

def erode(img: cv.Mat) :
    if(len(img.shape) == RGB_CHANNELS) :
        return Exception("Img need to be grayscale and binarzed")

    rows = img.shape[ROWS]
    cols = img.shape[COLS]

    result = img.copy()

    for i in range(1, rows - 1) :
        for j in range(1, cols - 1) :
            pixel = img[i,j]
            if not pixel == 255: continue

            if not img[i + 1, j] == 255: 
                result[i, j] = 0
                continue
            if not img[i, j + 1] == 255: 
                result[i, j] = 0
                continue
            if not img[i, j - 1] == 255: 
                result[i, j] = 0
                continue
            if not img[i - 1, j] == 255: 
                result[i, j] = 0
    
    return result

def opening(img: cv.Mat) :
    eroded = erode(erode(erode(img)))
    return dilate(dilate(dilate(eroded)))

def closing(img: cv.Mat) :
    dilated = dilate(dilate(dilate(img)))
    return erode(erode(erode(dilated)))

def homotopic_thinning(img: cv.Mat) :

    mask_1 = np.matrix([
        [0, 0, 0],
        [-1,1,-1],
        [1, 1, 1]
    ])

    mask_2 = np.matrix([
        [-1,  0,  0],
        [ 1,  1,  0],
        [ 1,  1, -1]
    ])

    mask_3 = np.matrix([
        [1, -1,  0],
        [1,  1,  0],
        [1, -1,  0]
    ])

    mask_4 = np.matrix([
        [ 1,  1, -1],
        [ 1,  1,  0],
        [-1,  0,  0]
    ])

    mask_5 = np.matrix([
        [ 1,  1,  1],
        [-1, 1, -1],
        [ 0,  0,  0]
    ])

    mask_6 = np.matrix([
        [-1, 1, 1],
        [0, 1, 1],
        [0, 0, -1]
    ])

    mask_7 = np.matrix([
        [0, -1, 1],
        [0, 1, 1],
        [0, -1, 1]
    ])

    mask_8 = np.matrix([
        [0, 0, -1],
        [0,  1, 1],
        [-1, 1, 1]
    ])

    masks = [mask_1, mask_2, mask_3, mask_4, mask_5, mask_6, mask_7, mask_8]

    result = img.copy()

    rows = img.shape[ROWS]
    cols = img.shape[COLS]

    nb_deleted = 1

    while nb_deleted > 0 :

        print(nb_deleted)

        nb_deleted = 0

        for i in range(2, rows - 1) :
            for j in range(2, cols - 1) :

                to_do = result[i, j] == 255

                if to_do :

                    to_delete = False
                    
                    for k in range(len(masks)) :
                        mask = masks[k]
                        
                        rows_mask = mask.shape[ROWS]
                        cols_mask = mask.shape[COLS]

                        to_delete_temp = True

                        for i_m in range(rows_mask) :
                            for j_m in range(cols_mask) :
                                mask_value = mask[i_m, j_m]

                                if mask_value == -1 : continue

                                diff_i = i_m - 1
                                diff_j = j_m - 1

                                if result[i + diff_i, j + diff_j] != (mask[i_m, j_m] * 255) :
                                    to_delete_temp = False
                                    break

                            if not to_delete_temp :
                                break
                        
                        if to_delete_temp :
                            to_delete = True
                            break
                            
                    if to_delete : 
                        nb_deleted += 1
                        result[i, j] = 0
                        break

    return result



                