import cv2
import matplotlib.pyplot as plt
import numpy as np
import math
import os


def read_img(img_path):
    """
        Read grayscale image
        Inputs:
        img_path: str: image path
        Returns:
        img: cv2 image
    """
    return cv2.imread(img_path, 0)


def padding_img(img, filter_size=3):
    """
    The surrogate function for the filter functions.
    The goal of the function: replicate padding the image such that when applying the kernel with the size of filter_size, the padded image will be the same size as the original image.
    WARNING: Do not use the exterior functions from available libraries such as OpenCV, scikit-image, etc. Just do from scratch using function from the numpy library or functions in pure Python.
    Inputs:
        img: cv2 image: original image
        filter_size: int: size of square filter
    Return:
        padded_img: cv2 image: the padding image
    """
    img = np.array(img)
    padded_img = np.pad(img, filter_size//2, mode="edge")
    return padded_img
  # Need to implement here

def mean_filter(img, filter_size=3):
    """
    Smoothing image with mean square filter with the size of filter_size. Use replicate padding for the image.
    WARNING: Do not use the exterior functions from available libraries such as OpenCV, scikit-image, etc. Just do from scratch using function from the numpy library or functions in pure Python.
    Inputs:
        img: cv2 image: original image
        filter_size: int: size of square filter,
    Return:
        smoothed_img: cv2 image: the smoothed image with mean filter.
    """
    filter_size = filter_size//2
    kernel = np.ones((filter_size, filter_size), dtype = int)
    padded_img = padding_img(img)
    smoothed_img = np.zeros(img.shape)
    h, w = img.shape
    for i in range(filter_size, h + filter_size): 
        for j in range(filter_size, w + filter_size): 
            temp = padded_img[i - filter_size : i + filter_size + 1, j - filter_size : j + filter_size + 1]
            mean = np.mean(temp)
            smoothed_img[i - filter_size, j - filter_size] = mean
    smoothed_img = smoothed_img.astype(np.uint8)
    return smoothed_img

def median_filter(img, filter_size=3):
    """
        Smoothing image with median square filter with the size of filter_size. Use replicate padding for the image.
        WARNING: Do not use the exterior functions from available libraries such as OpenCV, scikit-image, etc. Just do from scratch using function from the numpy library or functions in pure Python.
        Inputs:
            img: cv2 image: original image
            filter_size: int: size of square filter
        Return:
            smoothed_img: cv2 image: the smoothed image with median filter.
    """
    filter_size = filter_size//2
    kernel = np.ones((filter_size, filter_size), dtype = int)
    padded_img = padding_img(img)
    smoothed_img = np.zeros(img.shape)
    h, w = img.shape
    for i in range(filter_size, h + filter_size): 
        for j in range(filter_size, w + filter_size): 
            temp = padded_img[i - filter_size : i + filter_size + 1, j - filter_size : j + filter_size + 1]
            median = np.median(temp)
            smoothed_img[i - filter_size, j - filter_size] = median
    smoothed_img = smoothed_img.astype(np.uint8)
    return smoothed_img

def psnr(gt_img, smooth_img):
    """
        Calculate the PSNR metric
        Inputs:
            gt_img: cv2 image: groundtruth image
            smooth_img: cv2 image: smoothed image
        Outputs:
            psnr_score: PSNR score
    """
    #Calculate MSE
    gt_img = np.array(gt_img)
    smooth_img = np.array(smooth_img)
    mse = np.square(np.subtract(gt_img,smooth_img)).mean()
    if (mse == 0):
        mse = 100
    #Calculate PSNR
    psnr = 10 * math.log10((255**2)/mse)
    return psnr
    # Need to implement here



def show_res(before_img, after_img):
    """
        Show the original image and the corresponding smooth image
        Inputs:
            before_img: cv2: image before smoothing
            after_img: cv2: corresponding smoothed image
        Return:
            None
    """
    plt.figure(figsize=(12, 9))
    plt.subplot(1, 2, 1)
    plt.imshow(before_img, cmap='gray')
    plt.title('Before')

    plt.subplot(1, 2, 2)
    plt.imshow(after_img, cmap='gray')
    plt.title('After')
    plt.show()


if __name__ == '__main__':
    img_noise = "hw2/ex1_images/noise.png" # <- need to specify the path to the noise image
    img_gt = "hw2/ex1_images/ori_img.png" # <- need to specify the path to the gt image
    img = read_img(img_noise)
    filter_size = 3

    # Mean filter
    mean_smoothed_img = mean_filter(img, filter_size)
    show_res(img, mean_smoothed_img)
    print('PSNR score of mean filter: ', psnr(img, mean_smoothed_img))

    # Median filter
    median_smoothed_img = median_filter(img, filter_size)
    show_res(img, median_smoothed_img)
    print('PSNR score of median filter: ', psnr(img, median_smoothed_img))

