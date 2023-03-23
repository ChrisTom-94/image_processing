import cv2 as cv
import numpy as np

ROWS = 0
COLS = 1
CHANNELS = 2

UINT_8_VALUES = 256

GS_CHANNELS = 1
RGB_CHANNELS = 3

# convert from rgb to grayscale
def rgb_to_grayscale(img: cv.Mat):
    rows = img.shape[ROWS]
    cols = img.shape[COLS]

    RED_CONTRIBUTION = 0.299
    GREEN_CONTRIBUTION = 0.587
    BLUE_CONTRIBUTION = 0.114

    gray_image = np.zeros((rows, cols), dtype=np.uint8)

    for i in range(rows):
        for j in range(cols):
            gray_image[i,j] = int(RED_CONTRIBUTION * img[i,j,2] + GREEN_CONTRIBUTION * img[i,j,1] + BLUE_CONTRIBUTION * img[i,j,0])
    
    return gray_image

# generate histogram for grayscale and color images
def histo(img: cv.Mat):
    img = img.astype(np.float16).copy()

    if len(img.shape) == RGB_CHANNELS:
        return histo_rgb(img)
    
    hist = np.zeros((UINT_8_VALUES,1), dtype=np.uint8)

    rows = img.shape[ROWS]
    cols = img.shape[COLS]

    for i in range(rows):
        for j in range(cols):
            hist[int(img[i,j])] += 1
    
    return hist

# generate histogram for color images
def histo_rgb(img: cv.Mat):
    hist = [np.zeros((UINT_8_VALUES,1), dtype=np.uint8), np.zeros((UINT_8_VALUES,1), dtype=np.uint8), np.zeros((UINT_8_VALUES,1), dtype=np.uint8)]

    rows = img.shape[ROWS]
    cols = img.shape[COLS]

    for c in range(img.shape[CHANNELS]):
        for i in range(rows):
            for j in range(cols):
                hist[c][img[i,j,c]] += 1
    
    return hist

# generate cumulated histogram for grayscale and color images
def cumulated_histogram (img: cv.Mat, normalize: bool = False):
    if len(img.shape) == RGB_CHANNELS:
        return cumulated_histogram_rgb(img)

    hist = histo(img)
    frequencies = np.zeros((UINT_8_VALUES,1), dtype=np.float64)
    cumulated = np.zeros((UINT_8_VALUES,1), dtype=np.float64)
    img = img.astype(np.float16).copy()

    for i in range(len(frequencies)):
        cumulated[i] = hist[i] + cumulated[i-1] if i > 0 else hist[i]

    if normalize:
        rows = img.shape[ROWS]
        cols = img.shape[COLS]
        cumulated = cumulated / (rows * cols)
    
    return cumulated

# generate cumulated histogram for color images
def cumulated_histogram_rgb (img: cv.Mat, normalize: bool = False):
    hist = histo_rgb(img)
    frequencies = [np.zeros((UINT_8_VALUES,1), dtype=np.float64), np.zeros((UINT_8_VALUES,1), dtype=np.float64), np.zeros((UINT_8_VALUES,1), dtype=np.float64)]
    cumulated = [np.zeros((UINT_8_VALUES,1), dtype=np.float64), np.zeros((UINT_8_VALUES,1), dtype=np.float64), np.zeros((UINT_8_VALUES,1), dtype=np.float64)]

    for c in range(img.shape[CHANNELS]):
        for i in range(len(frequencies[c])):
            cumulated[c][i] = hist[c][i] + cumulated[c][i-1] if i > 0 else hist[c][i]
    
    if normalize:
        rows = img.shape[ROWS]
        cols = img.shape[COLS]
        for c in range(img.shape[CHANNELS]):
            cumulated[c] = cumulated[c] / (rows * cols)

    return cumulated

# binarize grayscale image using threshold
def binarize (img: cv.Mat, threshold: int):

    if len(img.shape) == RGB_CHANNELS:
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    
    bin_img = np.zeros((img.shape[ROWS], img.shape[COLS]), dtype=np.uint8)

    rows = img.shape[ROWS]
    cols = img.shape[COLS]

    bin = lambda i, j: bin_img[i,j] if img[i,j] < threshold else 255

    for i in range(rows):
        for j in range(cols):
            bin_img[i,j] = bin(i,j)
    
    return bin_img

# invert image
def invert(img : cv.Mat):
    return (UINT_8_VALUES - 1) - img

# stretch the histogram of a grayscale image
def stretch_histogram(img: cv.Mat):
    img = img.astype(np.float32).copy()

    if len(img.shape) == RGB_CHANNELS:
        return stretch_histogram_rgb(img)
    
    min, max, _, _ = cv.minMaxLoc(img)

    a = 255.0 / (max - min)
    b = -(a * min)

    print(a, b)

    return np.clip(a * img + b, 0, 255).astype(np.uint8)

# stretch the histogram of a color image
def stretch_histogram_rgb(img: cv.Mat):
    img_yuv = cv.cvtColor(img, cv.COLOR_BGR2YUV)
    img_yuv[:,:,0] = stretch_histogram(img_yuv[:,:,0])
    result = cv.cvtColor(img_yuv, cv.COLOR_YUV2BGR)
    
    return result.astype(np.uint8)

# equalize the histogram of a grayscale image
def equalize_histogram(img: cv.Mat):
    if len(img.shape) == RGB_CHANNELS:
        return equalize_histogram_rgb(img)
    
    cumulated = cumulated_histogram(img, True)
    img = img.astype(np.float32).copy()

    rows = img.shape[ROWS]
    cols = img.shape[COLS]

    for i in range(rows):
        for j in range(cols):
            img[i,j] = cumulated[np.uint8(img[i,j])] * (UINT_8_VALUES)
    
    return img.astype(np.uint8)

# equalize the histogram of a color image
def equalize_histogram_rgb(img: cv.Mat):
    img_yuv = cv.cvtColor(img, cv.COLOR_BGR2YUV)
    img_yuv[:,:,0] = cv.equalizeHist(img_yuv[:,:,0])
    result = cv.cvtColor(img_yuv, cv.COLOR_YUV2BGR)
    
    return result.astype(np.uint8)

# get pixel neighborhood
def get_pixel_neighborhood_indices(img: cv.Mat, pixel: tuple, connectivity: int):
    
    rows = img.shape[ROWS]
    cols = img.shape[COLS]

    i = pixel[0]
    j = pixel[1]

    neighborhood = []

    if i > 0 : neighborhood.append((i-1,j))
    if j > 0 : neighborhood.append((i,j-1))
    if i < rows - 1 : neighborhood.append((i+1,j))
    if j < cols - 1 : neighborhood.append((i,j+1))

    if connectivity == 8:
        if i > 0 :
            if j > 0 : neighborhood.append((i-1,j-1))
            if j < cols - 1 : neighborhood.append((i-1,j+1))
        if i < rows - 1 :
            if j > 0 : neighborhood.append((i+1,j-1))
            if j < cols - 1 : neighborhood.append((i+1,j+1))
    
    return neighborhood

