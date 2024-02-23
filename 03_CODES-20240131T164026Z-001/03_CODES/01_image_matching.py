
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt


#! Brute-Force Matching with ORB Descriptors 
def ORB_matching(img1, img2):
    # Initiate ORB detector
    orb = cv.ORB_create()
    # find the keypoints and descriptors with ORB
    kp1, des1 = orb.detectAndCompute(img1,None)
    kp2, des2 = orb.detectAndCompute(img2,None)

    # create BFMatcher object
    bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
    # Match descriptors.
    matches = bf.match(des1,des2)
    # Sort them in the order of their distance.
    matches = sorted(matches, key = lambda x:x.distance)
    # Draw first 10 matches.
    img3 = cv.drawMatches(img1,kp1,img2,kp2,matches[:10],None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    return img3


#! Brute-Force Matching with SIFT Descriptors 
def SIFT_matching(img1, img2):
    # Initiate SIFT detector
    sift = cv.SIFT_create()
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)
    # BFMatcher with default params
    bf = cv.BFMatcher()
    matches = bf.knnMatch(des1,des2,k=2)
    # Apply ratio test
    good = []
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            good.append([m])
    # cv.drawMatchesKnn expects list of lists as matches.
    img3 = cv.drawMatchesKnn(img1,kp1,img2,kp2,good,None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    return img3


#! FLANN based Matcher 
def FLANN_matching(img1, img2):
    # Initiate SIFT detector
    sift = cv.SIFT_create()
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)
    # FLANN parameters
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)   # or pass empty dictionary
    flann = cv.FlannBasedMatcher(index_params,search_params)
    matches = flann.knnMatch(des1,des2,k=2)
    # Need to draw only good matches, so create a mask
    matchesMask = [[0,0] for i in range(len(matches))]
    # ratio test as per Lowe's paper
    for i,(m,n) in enumerate(matches):
        if m.distance < 0.7*n.distance:
            matchesMask[i]=[1,0]
    draw_params = dict(matchColor = (0,255,0),
                    singlePointColor = (255,0,0),
                    matchesMask = matchesMask,
                    flags = cv.DrawMatchesFlags_DEFAULT)
    img3 = cv.drawMatchesKnn(img1,kp1,img2,kp2,matches,None,**draw_params)

    return img3


# Path of the images
input_img_rgb = r"D:\masters\Iaac barcelona\AT barcelona\02 software -1_2\Software_2.1_sem2\03_CODES-20240131T164026Z-001\n_iaac_01.png"
input_img_ir = r"D:\masters\Iaac barcelona\AT barcelona\02 software -1_2\Software_2.1_sem2\03_CODES-20240131T164026Z-001\n_iaac_02.png"

img1 = cv.imread(input_img_rgb,cv.IMREAD_GRAYSCALE) # queryImage
img2 = cv.imread(input_img_ir,cv.IMREAD_GRAYSCALE) # trainImage

# img3 = ORB_matching(img1, img2)
# img3 = SIFT_matching(img1, img2)
img3 = FLANN_matching(img1, img2)

# Display the image
plt.imshow(img3)
plt.show()

# Save the image
# cv.imwrite(r'output\img_match_RGB2IR_ORB_02.png',img3)
