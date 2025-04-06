#!/usr/bin/env python

import cv2
import numpy as np

def crop_image(img_path:str)->str:
    # load image
    img = cv2.imread(img_path)
    img = cv2.fastNlMeansDenoisingColored(img,None,10,10,7,21)
    img = img[:,:,:3]
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

    #Extra Cropping to bottom left corner
    roi=img[y+h-500:y+h,x:x+w-1400]
    cv2.imwrite('./debug/Image_crop2.jpg', roi)

    cropped_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) # convert to grayscale
    # threshold to get just the signature (INVERTED)
    retval, cropped_thresh_gray = cv2.threshold(cropped_gray, thresh=80, maxval=255, \
                                    type=cv2.THRESH_BINARY_INV)
    cv2.imwrite('./debug/Image_crop2_gray.jpg', cropped_thresh_gray)  # debugging
    

    # Draw bounding box rectangle (debugging)
    cv2.rectangle(img,(x,y),(x+w,y+h),(200,0,0),2)
    cv2.imwrite('./debug/Image_cont.jpg', img)
    return "./debug/Image_crop2_gray.jpg"

def testing():
    # Test the crop_image function
    cropped_image_path = crop_image('./test/test1.jpg')
    print(f"Cropped image saved at: {cropped_image_path}")


if __name__ == "__main__":
    testing()