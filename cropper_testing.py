#!/usr/bin/env python

import cv2
import numpy as np

# load image
img = cv2.imread('./test/test3.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # convert to grayscale
# threshold to get just the signature (INVERTED)
retval, thresh_gray = cv2.threshold(gray, thresh=160, maxval=255, \
                                   type=cv2.THRESH_BINARY_INV)
# Invert image
thresh_gray = (255-thresh_gray)
cv2.imwrite('./debug/Image_gray.jpg', thresh_gray)  # debugging

contours, hierarchy = cv2.findContours(thresh_gray,cv2.RETR_LIST, \
                                   cv2.CHAIN_APPROX_SIMPLE)

# Find object with the biggest bounding box
mx = (0,0,0,0)      # biggest bounding box so far
mx_area = 0
for cont in contours:
    x,y,w,h = cv2.boundingRect(cont)
    area = w*h
    if area > mx_area:
        mx = x,y,w,h
        mx_area = area
x,y,w,h = mx

# Crop and save
roi=img[y:y+h,x:x+w]
cv2.imwrite('./debug/Image_crop.jpg', roi)

# Draw bounding box rectangle (debugging)
cv2.rectangle(img,(x,y),(x+w,y+h),(200,0,0),2)
cv2.imwrite('./debug/Image_cont.jpg', img)