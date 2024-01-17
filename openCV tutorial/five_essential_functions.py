import cv2 as cv

img = cv.imread('Photos/cat_in_stop.jpg')

cv.imshow('Cat', img)

# 1: Converting into grayscale

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Gray', gray)

# 2: Blur
# Gaussian blur
blur = cv.GaussianBlur(img, (3, 3), cv.BORDER_DEFAULT) # accepts odd numbers only
cv.imshow('Blur', blur)

# 3: Edge Cascade
# Canny Edges
canny = cv.Canny(blur, 125, 175)
cv.imshow('Canny Edges', canny)

# 4: Dilating the image
dilated = cv.dilate(canny, (3,3), iterations=1)
cv.imshow('Dilated', dilated)

# Eroding : opposite of dilating
eroded = cv.erode(dilated, (3, 3), iterations=1)
cv.imshow('eroded', eroded)

# 5: Resize

resize = cv.resize(img, (500, 500))
cv.imshow('resized', resize)

# Cropping
cropped = img[50:200, 200:400]

cv.imshow('Cropped', cropped)


cv.waitKey(0)
