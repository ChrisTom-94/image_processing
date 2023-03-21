import numpy as np
import cv2 as cv
from utils import ROWS, COLS, CHANNELS, RGB_CHANNELS

def uniform_noise(img: cv.Mat, a: float, b: float):
    if img.shape[CHANNELS] == RGB_CHANNELS:
        return Exception('RGB images are not supported yet.')

    rows = img.shape[ROWS]
    cols = img.shape[COLS]

    noise = a + (b - a) * np.random.random((rows, cols))
    noisy_image = img + noise

    return noisy_image

def gaussian_noise(img: cv.Mat, mean: float, sigma: float):
    if img.shape[CHANNELS] == RGB_CHANNELS:
        return Exception('RGB images are not supported yet.')

    rows = img.shape[ROWS]
    cols = img.shape[COLS]

    noise = np.random.normal(mean, sigma, (rows, cols))
    noisy_image = img + noise

    return noisy_image

def salt_and_pepper_noise(img: cv.Mat, p: float):
    if img.shape[CHANNELS] == RGB_CHANNELS:
        return Exception('RGB images are not supported yet.')

    rows = img.shape[ROWS]
    cols = img.shape[COLS]

    noisy_image = img.copy()

    for i in range(rows):
        for j in range(cols):
            r = np.random.random()
            if r < p/2:
                noisy_image[i,j] = 0
            elif r < p:
                noisy_image[i,j] = 255

    return noisy_image

def speckle_noise(img: cv.Mat, sigma: float):
    if img.shape[CHANNELS] == RGB_CHANNELS:
        return Exception('RGB images are not supported yet.')

    rows = img.shape[ROWS]
    cols = img.shape[COLS]

    noise = np.random.normal(0, sigma, (rows, cols))
    noisy_image = img + img * noise

    return noisy_image

def poison_noise(img: cv.Mat, l: float):
    if img.shape[CHANNELS] == RGB_CHANNELS:
        return Exception('RGB images are not supported yet.')

    rows = img.shape[ROWS]
    cols = img.shape[COLS]

    noisy_image = np.zeros((rows, cols), dtype=np.uint8)

    for i in range(rows):
        for j in range(cols):
            noisy_image[i,j] = np.random.poisson(img[i,j] * l) / l

    return noisy_image

def exponential_noise(img: cv.Mat, l: float):
    if img.shape[CHANNELS] == RGB_CHANNELS:
        return Exception('RGB images are not supported yet.')

    rows = img.shape[ROWS]
    cols = img.shape[COLS]

    noisy_image = np.zeros((rows, cols), dtype=np.uint8)

    noise = np.random.normal(l, (rows, cols))

    for i in range(rows):
        for j in range(cols):
            noisy_image[i,j] = img[i,j] + noise[i,j]

    return noisy_image

def median_square_error(noised_image: cv.Mat, original_image: cv.Mat):
    if noised_image.shape[CHANNELS] == RGB_CHANNELS:
        return Exception('RGB images are not supported yet.')

    rows = noised_image.shape[ROWS]
    cols = noised_image.shape[COLS]

    mse = 0

    for i in range(rows):
        for j in range(cols):
            mse += (noised_image[i,j] - original_image[i,j]) ** 2

    mse /= (rows * cols)

    return mse

def peak_signal_to_noise_ratio(noised_image: cv.Mat, original_image: cv.Mat):  
    if noised_image.shape[CHANNELS] == RGB_CHANNELS:
        return Exception('RGB images are not supported yet.')

    mse = median_square_error(noised_image, original_image)

    return 10 * np.log10(255 ** 2 / mse)
