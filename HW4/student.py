import time
from math import floor
import numpy as np
import cv2
from scipy.sparse import csr_matrix

from scipy import ndimage, linalg

import util_sweep


def compute_photometric_stereo_impl(lights, images):
    """
    Given a set of images taken from the same viewpoint and a corresponding set
    of directions for light sources, this function computes the albedo and
    normal map of a Lambertian scene.

    If the computed albedo for a pixel has an L2 norm less than 1e-7, then set
    the albedo to black and set the normal to the 0 vector.

    Normals should be unit vectors.

    Input:
        lights -- N x 3 array.  Rows are normalized and are to be interpreted
                  as lighting directions.
        images -- list of N images.  Each image is of the same scene from the
                  same viewpoint, but under the lighting condition specified in
                  lights.
    Output:
        albedo -- float32 image. When the input 'images' are RGB, it should be of dimension height x width x 3,
                  while in the case of grayscale 'images', the dimension should be height x width x 1.
        normals -- float32 height x width x 3 image with dimensions matching
                   the input images.
    """

    # do not use nested for loops around weight, height and channels
    # use vectorization instead to make the computation much faster

    # shape_l = np.shape(lights)
    N, height, width, channels = np.shape(images)

    rshp_images = np.reshape(images, (N,height * width * channels))

    # G = np.dot(np.linalg.inv(np.dot(lights.T, lights)), np.dot(lights.T, I))
    # kd = np.linalg.norm(G)

    # LLinv =  np.linalg.inv(np.dot(lights, np.transpose(lights)))
    # LLinv_t_L = np.dot(LLinv, lights)
    # G = np.dot(LLinv_t_L,rshp_images)

    G_new = np.dot(np.linalg.inv(np.dot(lights.T, lights)), np.dot(lights.T, rshp_images))

    # color images
    color = np.reshape(G_new.T,(height,width,channels,3))
    albedo = np.linalg.norm(color , axis = 3)
    
    # grayscale images
    grayscale = np.mean(color, axis=2)
    albedo_for_norm = np.linalg.norm(grayscale, axis = 2)
    bools = albedo_for_norm < 1e-7
    
    normals = grayscale/np.maximum(1e-7, albedo_for_norm[:,:,np.newaxis])
    normals[bools]=0
    
    return albedo, normals

def project_impl(K, Rt, points):
    """
    Project 3D points into a calibrated camera.
    Input:
        K -- camera intrinsics calibration matrix
        Rt -- 3 x 4 camera extrinsics calibration matrix
        points -- height x width x 3 array of 3D points
    Output:
        projections -- height x width x 2 array of 2D projections
    """
    height = points.shape[0]
    width = points.shape[1]
    projections = np.zeros((height, width, 2))

    M = K.dot(Rt)

    for i in range(height):
        for j in range(width):

            p = np.append(points[i, j], 1)
            p = M.dot(p)
            projections[i, j, 0] = p[0] / p[2]
            projections[i, j, 1] = p[1] / p[2]
    
    return projections

def preprocess_ncc_impl(image, ncc_size):
    """
    Prepare normalized patch vectors according to normalized cross
    correlation.

    This is a preprocessing step for the NCC pipeline.  It is expected that
    'preprocess_ncc' is called on every input image to preprocess the NCC
    vectors and then 'compute_ncc' is called to compute the dot product
    between these vectors in two images.

    NCC preprocessing has two steps.
    (1) Compute and subtract the mean.
    (2) Normalize the vector.

    The mean is per channel.  i.e. For an RGB image, over the ncc_size**2
    patch, compute the R, G, and B means separately.  The normalization
    is over all channels.  i.e. For an RGB image, after subtracting out the
    RGB mean, compute the norm over the entire (ncc_size**2 * channels)
    vector and divide.

    If the norm of the vector is < 1e-6, then set the entire vector for that
    patch to zero.

    Patches that extend past the boundary of the input image at all should be
    considered zero.  Their entire vector should be set to 0.

    Patches are to be flattened into vectors with the default numpy row
    major order.  For example, given the following
    2 (height) x 2 (width) x 2 (channels) patch, here is how the output
    vector should be arranged.

    channel1         channel2
    +------+------+  +------+------+ height
    | x111 | x121 |  | x112 | x122 |  |
    +------+------+  +------+------+  |
    | x211 | x221 |  | x212 | x222 |  |
    +------+------+  +------+------+  v
    width ------->

    v = [ x111, x121, x211, x221, x112, x122, x212, x222 ]

    see order argument in np.reshape

    Input:
        image -- height x width x channels image of type float32
        ncc_size -- integer width and height of NCC patch region; assumed to be odd
    Output:
        normalized -- heigth x width x (channels * ncc_size**2) array
    """

   



def compute_ncc_impl(image1, image2):
    """
    Compute normalized cross correlation between two images that already have
    normalized vectors computed for each pixel with preprocess_ncc.

    Input:
        image1 -- height x width x (channels * ncc_size**2) array
        image2 -- height x width x (channels * ncc_size**2) array
    Output:
        ncc -- height x width normalized cross correlation between image1 and
               image2.
    """
    height, width, channels = image1.shape

    ncc1 = np.multiply(image1.reshape(height*width, -1), image2.reshape(height*width, -1))
    ncc2 = np.sum(ncc1, axis = 1)
    ncc = ncc2.reshape(height,width)

    return ncc

