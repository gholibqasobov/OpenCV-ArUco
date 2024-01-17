import cv2 as cv
import numpy as np
img = cv.imread('Photos/cat_in_stop.jpg')

cv.imshow('cat', img)

# Translation


def translate(img, x, y):
    transMat = np.float32([[1, 0, x], [0, 1, y]])
    dimensions = (img.shape[1], img.shape[0])
    return cv.warpAffine(img, transMat, dimensions)


"""
-x --> left
-y --> up
x --> right
y --> down
"""
translated = translate(img, -100, 100)
cv.imshow('tranlated', translated)


# Rotation
def rotate(img, angle, rotPoint=None):
    (height, width) = img.shape[:2]

    if rotPoint is None:
        rotPoint = (width//2, height//2)

    rotMat = cv.getRotationMatrix2D(rotPoint, angle, 1)
    dimensions = (width, height)
    return cv.warpAffine(img, rotMat, dimensions)


rotated = rotate(img, -90)
cv.imshow('rotated', rotated)

# Resizing
resize = cv.resize(img, (500, 500), interpolation=cv.INTER_LINEAR)
cv.imshow('resized', resize)


# Flipping

"""
1 - over x axis
0 - over y axis
-1 - over x and axes
"""

flip_x = cv.flip(img, 1)
cv.imshow('x axis', flip_x)

flip_y = cv.flip(img, 0)
cv.imshow('y axis', flip_y)

flip_both = cv.flip(img, -1)
cv.imshow('x and y axes', flip_both)

# Cropping
cropped = img[100:200, 100:300]
cv.imshow('cropped', cropped)

cv.waitKey(0)
