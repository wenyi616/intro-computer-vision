import math
import random

import cv2
import numpy as np

eTranslate = 0
eHomography = 1


def computeHomography(f1, f2, matches, A_out=None):
    '''
    Input:
        f1 -- list of cv2.KeyPoint objects in the first image
        f2 -- list of cv2.KeyPoint objects in the second image
        matches -- list of cv2.DMatch objects
            DMatch.queryIdx: The index of the feature in the first image
            DMatch.trainIdx: The index of the feature in the second image
            DMatch.distance: The distance between the two features
        A_out -- ignore this parameter. If computeHomography is needed
                 in other TODOs, call computeHomography(f1,f2,matches)
    Output:
        H -- 2D homography (3x3 matrix)
        Takes two lists of features, f1 and f2, and a list of feature
        matches, and estimates a homography from image 1 to image 2 from the matches.
    '''
    num_matches = len(matches)

    # Dimensions of the A matrix in the homogenous linear
    # equation Ah = 0
    num_rows = 2 * num_matches
    num_cols = 9
    A_matrix_shape = (num_rows,num_cols)
    A = np.zeros(A_matrix_shape)

    for i in range(len(matches)):
        m = matches[i]
        (a_x, a_y) = f1[m.queryIdx].pt
        (b_x, b_y) = f2[m.trainIdx].pt

        #TODO 2
        #TODO-BLOCK-BEGIN
        a = 2 * i

        A[a][0] = a_x
        A[a][1] = a_y
        A[a][2] = 1
        A[a][3] = 0
        A[a][4] = 0
        A[a][5] = 0
        A[a][6] = -a_x * b_x
        A[a][7] = -a_y * b_x
        A[a][8] = -b_x

        A[a + 1][0] = 0
        A[a + 1][1] = 0
        A[a + 1][2] = 0
        A[a + 1][3] = a_x
        A[a + 1][4] = a_y
        A[a + 1][5] = 1
        A[a + 1][6] = -a_x * b_y
        A[a + 1][7] = -a_y * b_y
        A[a + 1][8] = -b_y
        #TODO-BLOCK-END

    U, s, Vt = np.linalg.svd(A)

    if A_out is not None:
        A_out[:] = A

    #s is a 1-D array of singular values sorted in descending order
    #U, Vt are unitary matrices
    #Rows of Vt are the eigenvectors of A^TA.
    #Columns of U are the eigenvectors of AA^T.

    #Homography to be calculated
    H = np.eye(3)

    # TODO 3: Fill the homography H with the appropriate elements of the SVD
    # TODO-BLOCK-BEGIN
    H = Vt[:][-1].reshape((3,3))
    # TODO-BLOCK-END

    return H

def alignPair(f1, f2, matches, m, nRANSAC, RANSACthresh):
    '''
    Input:
        f1 -- list of cv2.KeyPoint objects in the first image
        f2 -- list of cv2.KeyPoint objects in the second image
        matches -- list of cv2.DMatch objects
            DMatch.queryIdx: The index of the feature in the first image
            DMatch.trainIdx: The index of the feature in the second image
            DMatch.distance: The distance between the two features
        m -- MotionModel (eTranslate, eHomography)
        nRANSAC -- number of RANSAC iterations
        RANSACthresh -- RANSAC distance threshold

    Output:
        M -- inter-image transformation matrix
        Repeat for nRANSAC iterations:
            Choose a minimal set of feature matches.
            Estimate the transformation implied by these matches
            count the number of inliers.
        For the transformation with the maximum number of inliers,
        compute the least squares motion estimate using the inliers,
        and return as a transformation matrix M.
    '''

    # TODO 4
    # Helper functions include compute_homography, get_inliers, and least_squares_fit.
    
    #TODO-BLOCK-BEGIN
    
    for i in range(nRANSAC):
        # pure translations (m == eTranslation)
        if m == eTranslate:
            
            sample = random.sample( range(0,len(matches)-1) , 1)
            (x1, y1) = f1[matches[sample[0]].queryIdx].pt
            (x2, y2) = f2[matches[sample[0]].trainIdx].pt
            tmp = np.array([[1, 0, x1-x2], [0, 1, y1-y2], [0, 0, 1]])
        
        # full homographies (m == eHomography)
        elif m == eHomography:
            samples = []
            samples.append(random.sample( range(0,len(matches)-1) , 1))
            samples.append(random.sample( range(0,len(matches)-1) , 1))
            samples.append(random.sample( range(0,len(matches)-1) , 1))
            samples.append(random.sample( range(0,len(matches)-1) , 1))

            for sample in samples:
                new_match = []
                new_match.append(matches[sample[0]])

            tmp = computeHomography(f1,f2,new_match)
        
    max_in = -1
    inliers = []
        
    inlier = getInliers(f1, f2, matches, tmp, RANSACthresh)
    print(inlier)

    if  len(inlier) > max_in:
        max_in = len(inlier)
        best_inlier = inlier
    
    M = leastSquaresFit(f1, f2, matches, m, best_inlier)
    #TODO-BLOCK-END
    return M

def getInliers(f1, f2, matches, M, RANSACthresh):
    '''
    Input:
        f1 -- list of cv2.KeyPoint objects in the first image
        f2 -- list of cv2.KeyPoint objects in the second image
        matches -- list of cv2.DMatch objects
            DMatch.queryIdx: The index of the feature in the first image
            DMatch.trainIdx: The index of the feature in the second image
            DMatch.distance: The distance between the two features
        M -- inter-image transformation matrix
        RANSACthresh -- RANSAC distance threshold

    Output:
        inlier_indices -- inlier match indices (indexes into 'matches')

        Transform the matched features in f1 by M.
        Store the match index of features in f1 for which the transformed
        feature is within Euclidean distance RANSACthresh of its match
        in f2.
        Return the array of the match indices of these features.
    '''

    inlier_indices = []

    for i in range(len(matches)):
        #BEGIN TODO 5
        #Determine if the ith matched feature f1[id1], when transformed
        #by M, is within RANSACthresh of its match in f2.
        #If so, append i to inliers
        #TODO-BLOCK-BEGIN
        raise Exception("TODO in alignment.py not implemented")
        #TODO-BLOCK-END
        #END TODO

    return inlier_indices

def leastSquaresFit(f1, f2, matches, m, inlier_indices):
    '''
    Input:
        f1 -- list of cv2.KeyPoint objects in the first image
        f2 -- list of cv2.KeyPoint objects in the second image
        matches -- list of cv2.DMatch objects
            DMatch.queryIdx: The index of the feature in the first image
            DMatch.trainIdx: The index of the feature in the second image
            DMatch.distance: The distance between the two features
        m -- MotionModel (eTranslate, eHomography)
        inlier_indices -- inlier match indices (indexes into 'matches')

    Output:
        M - transformation matrix

        Compute the transformation matrix from f1 to f2 using only the
        inliers and return it.
    '''

    # This function needs to handle two possible motion models,
    # pure translations (eTranslate)
    # and full homographies (eHomography).

    M = np.eye(3)

    if m == eTranslate:
        #For spherically warped images, the transformation is a
        #translation and only has two degrees of freedom.
        #Therefore, we simply compute the average translation vector
        #between the feature in f1 and its match in f2 for all inliers.

        u = 0.0
        v = 0.0

        for i in range(len(inlier_indices)):
            #BEGIN TODO 6
            #Use this loop to compute the average translation vector
            #over all inliers.
            #TODO-BLOCK-BEGIN
            raise Exception("TODO in alignment.py not implemented")
            #TODO-BLOCK-END
            #END TODO

        u /= len(inlier_indices)
        v /= len(inlier_indices)

        M[0,2] = u
        M[1,2] = v

    elif m == eHomography:
        #BEGIN TODO 7
        #Compute a homography M using all inliers.
        #This should call computeHomography.
        #TODO-BLOCK-BEGIN
        raise Exception("TODO in alignment.py not implemented")
        #TODO-BLOCK-END
        #END TODO

    else:
        raise Exception("Error: Invalid motion model.")

    return M
