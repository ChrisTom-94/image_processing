import numpy as np
import cv2 as cv
from utils import get_pixel_neighborhood, RGB_CHANNELS, ROWS, COLS, CHANNELS

###########################
###### median filter ######
###########################

def median(img: np.ndarray, i: int, j: int, kernel_size: int):
    neighborhood = get_pixel_neighborhood((i,j), img.shape[ROWS], img.shape[COLS], kernel_size)
    values = [img[pixel] for pixel in neighborhood]
    values.append(img[i,j])
    values.sort()
    return values[len(values)//2]
    
def median_filter(img: cv.Mat, kernel_size: int):
    if kernel_size % 2 == 0:
        raise Exception('Kernel size must be odd')
    
    rows = img.shape[ROWS]
    cols = img.shape[COLS]

    if len(img.shape) == RGB_CHANNELS:
        return median_filter_rgb(img, rows, cols, kernel_size)
    
    img = img.astype(np.float16).copy()
    
    filtered = np.zeros((rows, cols), dtype=np.uint8)
    
    for i in range(rows):
        for j in range(cols):
            filtered[i,j] = median(img, i, j, kernel_size)
    
    return filtered

def median_filter_rgb(img: cv.Mat, rows: int, cols: int, kernel_size: int):
    filtered = np.zeros((rows, cols, 3), dtype=np.uint8)
    
    for c in range(img.shape[CHANNELS]):
        for i in range(rows):
            for j in range(cols):
                filtered[i,j,c] = median(img[:,:,c], i, j, kernel_size)
    
    return filtered


###########################
##### Gaussian filter #####
###########################

def gaussian_filter(img: cv.Mat, kernel_size: int, sigma: float):
    if kernel_size % 2 == 0:
        raise Exception('Kernel size must be odd')
    
    if len(img.shape) == RGB_CHANNELS:
        return Exception('Gaussian filter not implemented for RGB images')
    
    rows = img.shape[ROWS]
    cols = img.shape[COLS]

    img = img.astype(np.float16).copy()

    filtered = np.zeros((rows, cols), dtype=np.uint8)

    for i in range(rows):
        for j in range(cols):
            neighborhood = get_pixel_neighborhood((i,j), rows, cols, kernel_size)
            values = [img[pixel] for pixel in neighborhood]
            filtered[i,j] = gaussian_filter_pixel(img[i,j], values, sigma)

    return filtered


def gaussian_filter_pixel(pixel: tuple(int, int), neighborhood: list, sigma: float):
    sum = 0
    for i in range(len(neighborhood)):
        x = pixel[0] - neighborhood[i][0]
        y = pixel[1] - neighborhood[i][1]
        gaussian_value = 1/(2*np.pi*sigma**2) * np.exp(-(x**2+y**2)/(2*sigma**2))
        sum = sum + gaussian_value * neighborhood[i]
    
    return sum

###########################
#### non linear filter ####
###########################

def non_linear_filter(img: cv.Mat, kernel_size: int, filter_function: function):
    if kernel_size % 2 == 0:
        raise Exception('Kernel size must be odd')
    
    rows = img.shape[ROWS]
    cols = img.shape[COLS]

    if len(img.shape) == RGB_CHANNELS:
        return non_linear_filter_rgb(img, rows, cols, kernel_size, filter_function)
    
    img = img.astype(np.float16).copy()
    
    filtered = np.zeros((rows, cols), dtype=np.uint8)
    
    for i in range(rows):
        for j in range(cols):
            neighborhood = get_pixel_neighborhood((i,j), rows, cols, kernel_size)
            values = [img[pixel] for pixel in neighborhood]
            filtered[i,j] = filter_function(values)
    
    return filtered

def non_linear_filter_rgb(img: cv.Mat, rows: int, cols: int, kernel_size: int, filter_function):
    filtered = np.zeros((rows, cols, 3), dtype=np.uint8)
    
    for c in range(img.shape[CHANNELS]):
        for i in range(rows):
            for j in range(cols):
                neighborhood = get_pixel_neighborhood((i,j), rows, cols, kernel_size)
                values = [img[pixel,c] for pixel in neighborhood]
                filtered[i,j,c] = filter_function(values)
    
    return filtered
