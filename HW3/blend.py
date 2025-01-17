import math
import sys

import cv2
import numpy as np


class ImageInfo:
    def __init__(self, name, img, position):
        self.name = name
        self.img = img
        self.position = position


def imageBoundingBox(img, M):
    """
       This is a useful helper function that you might choose to implement
       that takes an image, and a transform, and computes the bounding box
       of the transformed image.

       INPUT:
         img: image to get the bounding box of
         M: the transformation to apply to the img
       OUTPUT:
         minX: int for the minimum X value of a corner
         minY: int for the minimum Y value of a corner
         minX: int for the maximum X value of a corner
         minY: int for the maximum Y value of a corner
    """
    #TODO 8
    h, w = img.shape[:2]
    pts = np.array([[0, 0, 1], [0, h-1, 1], [w-1, 0, 1], [w-1, h-1, 1]]).T
    
    res = np.dot(M, pts)
    res = res / res[-1]
    minX, minY, _ = np.min(res, axis=1)
    maxX, maxY, _ = np.max(res, axis=1)
    # TODO-BLOCK-END

    return int(minX), int(minY), int(maxX), int(maxY)


def accumulateBlend(img, acc, M, blendWidth):
    """
       INPUT:
         img: image to add to the accumulator
         acc: portion of the accumulated image where img should be added
         M: the transformation mapping the input image to the accumulator
         blendWidth: width of blending function. horizontal hat function
       OUTPUT:
         modify acc with weighted copy of img added where the first
         three channels of acc record the weighted sum of the pixel colors
         and the fourth channel of acc records a sum of the weights
    """
    # BEGIN TODO 10
    r, c = acc.shape[:2]
    rows, cols = img.shape[:2]
    img = cv2.copyMakeBorder(img, 0, r - rows, 0, c - cols, cv2.BORDER_CONSTANT, value=0)

    row, col, _ = img.shape
    x_range = np.arange(col)
    y_range = np.arange(row)

    x_grid, y_grid = np.meshgrid(x_range, y_range)
    ones = np.ones((row, col))
    
    # homogeneous coordinates
    homoco = np.dstack((x_grid, y_grid, ones))
    homoco = homoco.reshape((col * row, 3))
    homoco = homoco.T
    projection = np.linalg.inv(M).dot(homoco) 

    # convert back to img plane
    projection = projection / projection[2]
    x = projection[0].reshape((row, col)).astype(np.float32)
    y = projection[1].reshape((row, col)).astype(np.float32)

    minX, minY, maxX, maxY = imageBoundingBox(img, M)

    img_warped = cv2.remap(img, x, y, cv2.INTER_LINEAR)
    img_masked = np.dstack((img_warped, np.ones((img_warped.shape[0], img_warped.shape[1], 1)) ))

    base = np.linspace(-minX / blendWidth, (c - minX -1) / blendWidth, c )

    # feathering: linear interpolation
    right = np.clip(base , 0, 1).reshape((1, c, 1))
    left = np.ones((1, c, 1)) - right

    feathered_img = right * img_masked
    acc *= left

    grayimg = cv2.cvtColor(img_warped, cv2.COLOR_BGR2GRAY)
    maskimg = (grayimg != 0).reshape((r, c, 1))
    img_masked = maskimg * feathered_img

    grayacc = cv2.cvtColor(acc[:, :, :3].astype(np.uint8), cv2.COLOR_BGR2GRAY)
    maskacc = (grayacc != 0).reshape((r, c, 1))
    
    acc *= maskacc
    acc += img_masked
    
    # END TODO


def normalizeBlend(acc):
    """
       INPUT:
         acc: input image whose alpha channel (4th channel) contains
         normalizing weight values
       OUTPUT:
         img: image with r,g,b values of acc normalized
    """
    # BEGIN TODO 11
    h, w = acc.shape[:2]

    img = np.zeros((h, w, 3))
    for i in range(h):
        for j in range(w):
            weights = acc[i, j, 3]
            if weights > 0:
                img[i, j, :] = acc[i, j, :3] / weights

    img = np.uint8(img)
    # END TODO
    return img


def getAccSize(ipv):
    """
       This function takes a list of ImageInfo objects consisting of images and
       corresponding transforms and Returns useful information about the accumulated
       image.

       INPUT:
         ipv: list of ImageInfo objects consisting of image (ImageInfo.img) and transform(image (ImageInfo.position))
       OUTPUT:
         accWidth: Width of accumulator image(minimum width such that all tranformed images lie within acc)
         accWidth: Height of accumulator image(minimum height such that all tranformed images lie within acc)

         channels: Number of channels in the accumulator image
         width: Width of each image(assumption: all input images have same width)
         translation: transformation matrix so that top-left corner of accumulator image is origin
    """

    # Compute bounding box for the mosaic
    minX = np.Inf
    minY = np.Inf
    maxX = 0
    maxY = 0
    channels = -1
    width = -1  # Assumes all images are the same width
    M = np.identity(3)
    for i in ipv:
        M = i.position
        img = i.img
        _, w, c = img.shape
        if channels == -1:
            channels = c
            width = w

        # TODO 9: update minX, ..., maxY
        boundings = imageBoundingBox(img, M)

        minX = min(minX, boundings[0])
        minY = min(minY, boundings[1])
        maxX = max(maxX, boundings[2])
        maxY = max(maxY, boundings[3])
        # END TODO

    # Create an accumulator image
    accWidth = int(math.ceil(maxX) - math.floor(minX))
    accHeight = int(math.ceil(maxY) - math.floor(minY))

    translation = np.array([[1, 0, -minX], [0, 1, -minY], [0, 0, 1]])

    return accWidth, accHeight, channels, width, translation


def pasteImages(ipv, translation, blendWidth, accWidth, accHeight, channels):
    acc = np.zeros((accHeight, accWidth, channels + 1))
    # Add in all the images
    M = np.identity(3)
    for count, i in enumerate(ipv):
        M = i.position
        img = i.img

        M_trans = translation.dot(M)
        accumulateBlend(img, acc, M_trans, blendWidth)

    return acc


def getDriftParams(ipv, translation, width):
    # Add in all the images
    M = np.identity(3)
    for count, i in enumerate(ipv):
        if count != 0 and count != (len(ipv) - 1):
            continue

        M = i.position

        M_trans = translation.dot(M)

        p = np.array([0.5 * width, 0, 1])
        p = M_trans.dot(p)

        # First image
        if count == 0:
            x_init, y_init = p[:2] / p[2]
        # Last image
        if count == (len(ipv) - 1):
            x_final, y_final = p[:2] / p[2]

    return x_init, y_init, x_final, y_final


def computeDrift(x_init, y_init, x_final, y_final, width):
    A = np.identity(3)
    drift = (float)(y_final - y_init)
    # We implicitly multiply by -1 if the order of the images is swapped...
    length = (float)(x_final - x_init)
    A[0, 2] = -0.5 * width
    # Negative because positive y points downwards
    A[1, 0] = -drift / length

    return A


def blendImages(ipv, blendWidth, is360=False, A_out=None):
    """
       INPUT:
         ipv: list of input images and their relative positions in the mosaic
         blendWidth: width of the blending function
       OUTPUT:
         croppedImage: final mosaic created by blending all images and
         correcting for any vertical drift
    """
    accWidth, accHeight, channels, width, translation = getAccSize(ipv)
    acc = pasteImages(
        ipv, translation, blendWidth, accWidth, accHeight, channels
    )
    compImage = normalizeBlend(acc)

    # Determine the final image width
    outputWidth = (accWidth - width) if is360 else accWidth
    x_init, y_init, x_final, y_final = getDriftParams(ipv, translation, width)
    # Compute the affine transform
    A = np.identity(3)
    # TODO 12

    # Note: warpPerspective does forward mapping which means A is an affine
    # transform that maps accumulator coordinates to final panorama coordinates
    # if this is a 360 panorama

    if is360:
        A = computeDrift(x_init, y_init, x_final, y_final, width)
    # END TODO

    if A_out is not None:
        A_out[:] = A

    # Warp and crop the composite
    croppedImage = cv2.warpPerspective(
        compImage, A, (outputWidth, accHeight), flags=cv2.INTER_LINEAR
    )

    return croppedImage
