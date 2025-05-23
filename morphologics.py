import numpy as np
import cv2 as cv
from utils import ROWS, COLS, RGB_CHANNELS

BLACK = 0
WHITE = 255
JOKER = -1
LOWER_LIMIT = 0

def dilate(img: cv.Mat, mask: cv.Mat, mask_kernel: tuple, iterations: int = 1, get_added: bool = False) :
    
    if(len(img.shape) == RGB_CHANNELS) :
        return Exception("Image need to be in grayscale and binarized")
    
    rows_img = img.shape[ROWS]
    cols_img = img.shape[COLS]

    non_zeros_mask = cv.findNonZero(mask)
    non_zero_mask_length = len(non_zeros_mask)
    
    result = img.copy()
    added = []

    for _ in range(iterations) :

        non_zeros_img = cv.findNonZero(result)

        if non_zeros_img is None :
            break

        non_zeros_img_length = len(non_zeros_img)

        for i in range(non_zeros_img_length) :
            # findnonzero return a list with inverted coordinates
            # so COLS is the row and ROWS is the column
            row_img = non_zeros_img[i][0][COLS]
            col_img = non_zeros_img[i][0][ROWS]

            for j in range(non_zero_mask_length) :
                # same as above, coordinates are inverted
                row_mask = non_zeros_mask[j][0][COLS]
                col_mask = non_zeros_mask[j][0][ROWS]

                diff_i = row_mask - mask_kernel[ROWS]
                diff_j = col_mask - mask_kernel[COLS]

                neighbor = [row_img + diff_i, col_img + diff_j]

                if neighbor[ROWS] < LOWER_LIMIT or neighbor[ROWS] >= rows_img : continue
                if neighbor[COLS] < LOWER_LIMIT or neighbor[COLS] >= cols_img : continue

                if not result[neighbor[ROWS], neighbor[COLS]] == WHITE :
                    result[neighbor[ROWS], neighbor[COLS]] = WHITE
                    added.append((neighbor[ROWS], neighbor[COLS]))

    if get_added : 
        return [result, added]
    else :
        return result  
            

def erode(img: cv.Mat, mask: cv.Mat, mask_kernel: tuple, iterations: int = 1, get_removed: bool = False) : 
    if(len(img.shape) == RGB_CHANNELS) :
        return Exception("Image need to be in grayscale and binarized")

    rows = img.shape[ROWS]
    cols = img.shape[COLS]

    non_zeros_mask = cv.findNonZero(mask)
    non_zero_mask_length = len(non_zeros_mask)

    result = img.copy()
    removed = []

    for _ in range(iterations) :

        non_zeros_img = cv.findNonZero(result)

        if non_zeros_img is None :
            break

        non_zeros_img_length = len(non_zeros_img)

        tmp_result = result.copy()

        for i in range(non_zeros_img_length) :
            # findnonzero return a list with inverted coordinates
            # so COLS is the row and ROWS is the column
            row_img = non_zeros_img[i][0][COLS]
            col_img = non_zeros_img[i][0][ROWS]

            for j in range(non_zero_mask_length) :
                # same as above, coordinates are inverted
                row_mask = non_zeros_mask[j][0][COLS]
                col_mask = non_zeros_mask[j][0][ROWS]

                diff_i = row_mask - mask_kernel[ROWS]
                diff_j = col_mask - mask_kernel[COLS]

                neighbor = [row_img + diff_i, col_img + diff_j]

                if neighbor[ROWS] < LOWER_LIMIT or neighbor[ROWS] >= rows : continue
                if neighbor[COLS] < LOWER_LIMIT or neighbor[COLS] >= cols : continue

                if not tmp_result[neighbor[ROWS], neighbor[COLS]] == WHITE :
                    result[row_img, col_img] = BLACK
                    removed.append((row_img, col_img))
                    break
    
    if get_removed :
        return [result, removed]
    else :
        return result
    

def opening(img: cv.Mat, mask: cv.Mat, mask_kernel: tuple, iterations: int = 1) :
    eroded = erode(img, mask, mask_kernel, iterations)
    return dilate(eroded, mask, mask_kernel, iterations)

def closing(img: cv.Mat, mask: cv.Mat, mask_kernel: tuple, iterations: int = 1) :
    dilated = dilate(img, mask, mask_kernel, iterations)
    return erode(dilated, mask, mask_kernel, iterations)

def rotate_matrix_by_45(matrix: np.matrix, rotate: int = 1) :
    rows = matrix.shape[ROWS]
    cols = matrix.shape[COLS]

    result = matrix.copy()

    for _ in range(rotate) :
        for i in range(rows) :
            for j in range(cols) :
                if i not in [0, rows - 1] :
                    if j not in [0, cols - 1] : continue
                    if j == 0 :
                        result[i - 1, j] = matrix[i, j]
                    else :
                        result[i + 1, j] = matrix[i, j]
                elif i == rows - 1 :
                    if j == 0 :
                        result[i - 1, j] = matrix[i, j]
                    else : 
                        result[i, j - 1] = matrix[i, j]
                else :
                    if j == cols - 1 :
                        result[i + 1, j] = matrix[i, j]
                    else :
                        result[i, j + 1] = matrix[i, j]
    return result

def homotopic_thinning(img: cv.Mat, mask: cv.Mat) :
    if(len(img.shape) == RGB_CHANNELS) :
        return Exception("Image need to be in grayscale and binarized")

    rows = img.shape[ROWS]
    cols = img.shape[COLS]

    mask_rows = mask.shape[ROWS]
    mask_cols = mask.shape[COLS]
    mask_center_row = mask_rows // 2
    mask_center_col = mask_cols // 2

    result = img.copy()

    nb_deleted = 1

    while nb_deleted > 0 :

        non_zeros_img = cv.findNonZero(result)
        non_zeros_img_length = len(non_zeros_img)

        nb_deleted = 0

        tmp_result = result.copy()

        for k in range(non_zeros_img_length) :
            row_img = non_zeros_img[k][0][COLS]
            col_img = non_zeros_img[k][0][ROWS]

            to_delete = False

            tmp_mask = mask.copy()

            while True :

                to_delete_temp = True

                for row_mask in range(mask_rows) :
                    for col_mask in range(mask_cols) :
                        mask_value = tmp_mask[row_mask, col_mask]
                        if mask_value == JOKER : continue

                        diff_row = row_mask - mask_center_row
                        diff_col = col_mask - mask_center_col

                        neighbor = [row_img + diff_row, col_img + diff_col]
                        if neighbor[ROWS] < LOWER_LIMIT or neighbor[ROWS] >= rows : continue
                        if neighbor[COLS] < LOWER_LIMIT or neighbor[COLS] >= cols : continue

                        if tmp_result[neighbor[ROWS], neighbor[COLS]] != (tmp_mask[row_mask, col_mask] * WHITE) :
                            to_delete_temp = False
                            break

                    if not to_delete_temp :
                        break
                
                if to_delete_temp :
                    to_delete = True
                    break

                tmp_mask = rotate_matrix_by_45(tmp_mask)

                if np.array_equal(tmp_mask, mask) :
                    break

                    
            if to_delete : 
                nb_deleted += 1
                result[row_img, col_img] = BLACK
        
        print(nb_deleted)

    return result


def lantuejoul(img: cv.Mat, mask: cv.Mat):
    if(len(img.shape) == RGB_CHANNELS) :
        return Exception("Image need to be in grayscale and binarized")

    rows = img.shape[ROWS]
    cols = img.shape[COLS]

    skeleton = np.zeros((rows, cols), dtype=np.uint8)
    recompose_function = np.zeros((rows, cols), dtype=np.uint8)

    tmp = img.copy()

    iter = 0
    while True :
        iter += 1
        eroded = erode(tmp, mask, (1,1))

        [eroded_opening, removed] = erode(eroded, mask, (1,1), get_removed=True)

        if len(removed) == 0 :
            break

        [opening, added] = dilate(eroded_opening, mask, (1,1), get_added=True)

        tmp = opening.copy()

        diff = list(set(removed) - set(added))

        for i, j in diff : 
            skeleton[i, j] = 255
            recompose_function[i, j] = iter


    return [skeleton, recompose_function]


def recompose_from_lantuejoul(skeleton: cv.Mat, recompose_function: cv.Mat) : 

    rows = skeleton.shape[ROWS]
    cols = skeleton.shape[COLS]

    non_zeros_img = cv.findNonZero(skeleton)
    non_zeros_length = len(non_zeros_img)

    recomposed = np.zeros((rows, cols))

    for i in range(non_zeros_length):
        row = non_zeros_img[i][0][COLS]
        col = non_zeros_img[i][0][ROWS]

        tmp_skelet = np.zeros((rows, cols))
        tmp_skelet[row, col] = 255

        iter = recompose_function[row, col]

        mask = np.ones((3,3))

        result = dilate(tmp_skelet, mask, (1,1), iter)

        for x in range(rows) :
            for y in range(cols) :
                if result[x, y] == WHITE :
                    recomposed[x, y] = WHITE

    return recomposed





                