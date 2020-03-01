# Wenyi Chu

import sys
import cv2
import numpy as np

def cross_correlation_helper(img, kernel, m, n):
    h = img.shape[0]
    w = img.shape[1]
    new_img = np.zeros(img.shape)

    # loop thru each pixel in the image
    for x in range(h):
        for y in range(w):
            # compute the new intensity at position [x,y]

            # left upper corner: [a,b]
            a = x - (m-1)/2
            b = y - (n-1)/2
            
            new_intensity = 0

            for i in range(m):
                for j in range(n):

                    if a+i < 0 or a+i >= h or b+j < 0 or b+j >= w:
                        new_intensity += 0
                    else: 
                        new_intensity += img[a+i][b+j] * kernel[i][j]

            new_img[x][y] = new_intensity

    return new_img

def cross_correlation_2d(img, kernel):
    '''Given a kernel of arbitrary m x n dimensions, with both m and n being
    odd, compute the cross correlation of the given image with the given
    kernel, such that the output is of the same dimensions as the image and that
    you assume the pixels out of the bounds of the image to be zero. Note that
    you need to apply the kernel to each channel separately, if the given image
    is an RGB image.

    Inputs:
        img:    Either an RGB image (height x width x 3) or a grayscale image
                (height x width) as a numpy array.
        kernel: A 2D numpy array (m x n), with m and n both odd (but may not be
                equal).

    Output:
        Return an image of the same dimensions as the input image (same width,
        height and the number of color channels)
    '''
    # TODO-BLOCK-BEGIN  
    m = kernel.shape[0]
    n = kernel.shape[1]

    if len(img.shape) == 2:         
        new_img = cross_correlation_helper(img, kernel, m, n)
        return new_img
    
    else:
        img_height, img_width, image_channels = img.shape
        r,g,b = np.split(img,3,axis=2)
        new_img = cross_correlation_helper(r, kernel, m, n)
        new_img = np.dstack((new_img, cross_correlation_helper(g, kernel, m, n)))
        new_img = np.dstack((new_img, cross_correlation_helper(b, kernel, m, n)))
        return new_img

    return new_img
    raise Exception("TODO in hybrid.py not implemented")
    # TODO-BLOCK-END


def convolve_2d(img, kernel):
    '''Use cross_correlation_2d() to carry out a 2D convolution.

    Inputs:
        img:    Either an RGB image (height x width x 3) or a grayscale image
                (height x width) as a numpy array.
        kernel: A 2D numpy array (m x n), with m and n both odd (but may not be
                equal).
    Output:
        Return an image of the same dimensions as the input image (same width,
        height and the number of color channels)
    '''
    # TODO-BLOCK-BEGIN
    return cross_correlation_2d(img,np.flip(kernel))
    raise Exception("TODO in hybrid.py not implemented")
    # TODO-BLOCK-END


def gaussian_blur_kernel_2d(sigma, height, width):
    '''Return a Gaussian blur kernel of the given dimensions and with the given
    sigma. Note that width and height are different.

    Input:
        sigma:  The parameter that controls the radius of the Gaussian blur.
                Note that, in our case, it is a circular Gaussian (symmetric
                across height and width).
        width:  The width of the kernel.
        height: The height of the kernel.

    Output:
        Return a kernel of dimensions height x width such that convolving it
        with an image results in a Gaussian-blurred image.
    '''
    # TODO-BLOCK-BEGIN
    kernel = np.zeros(shape=(height, width))

    x = np.floor(width/2)
    y = np.floor(height/2)

    for i in range(width):
        for j in range(height):
            kernel[j, i] = (1 / (2 * np.pi * sigma ** 2)) * \
                np.exp(-1 * ((i-x) ** 2 + (j-y) ** 2) / (2 * sigma ** 2))
    
    # normalized
    return kernel/kernel.sum()
    raise Exception("TODO in hybrid.py not implemented")
    # TODO-BLOCK-END

def low_pass(img, sigma, size):
    '''Filter the image as if its filtered with a low pass filter of the given
    sigma and a square kernel of the given size. A low pass filter supresses
    the higher frequency components (finer details) of the image.

    Output:
        Return an image of the same dimensions as the input image (same width,
        height and the number of color channels)
    '''
    # TODO-BLOCK-BEGIN
    return convolve_2d(img,gaussian_blur_kernel_2d(sigma, size, size))
    raise Exception("TODO in hybrid.py not implemented")
    # TODO-BLOCK-END

def high_pass(img, sigma, size):
    '''Filter the image as if its filtered with a high pass filter of the given
    sigma and a square kernel of the given size. A high pass filter suppresses
    the lower frequency components (coarse details) of the image.

    Output:
        Return an image of the same dimensions as the input image (same width,
        height and the number of color channels)
    '''
    # TODO-BLOCK-BEGIN
    return img - low_pass(img,sigma,size)
    raise Exception("TODO in hybrid.py not implemented")
    # TODO-BLOCK-END

def create_hybrid_image(img1, img2, sigma1, size1, high_low1, sigma2, size2,
        high_low2, mixin_ratio, scale_factor):
    '''This function adds two images to create a hybrid image, based on
    parameters specified by the user.'''
    high_low1 = high_low1.lower()
    high_low2 = high_low2.lower()

    if img1.dtype == np.uint8:
        img1 = img1.astype(np.float32) / 255.0
        img2 = img2.astype(np.float32) / 255.0

    if high_low1 == 'low':
        img1 = low_pass(img1, sigma1, size1)
    else:
        img1 = high_pass(img1, sigma1, size1)

    if high_low2 == 'low':
        img2 = low_pass(img2, sigma2, size2)
    else:
        img2 = high_pass(img2, sigma2, size2)

    img1 *=  (1 - mixin_ratio)
    img2 *= mixin_ratio
    hybrid_img = (img1 + img2) * scale_factor
    return (hybrid_img * 255).clip(0, 255).astype(np.uint8)