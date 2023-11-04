import cv2 as cv
import numpy as np


def corner_harris(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY) 

    gray = np.float32(gray)
    cronerHarris = cv.cornerHarris(gray, 2, 3, 0.04)

    image[cronerHarris > 0.01*cronerHarris.max()] = [255, 0, 0]
    return image

def surf(image):
    orb = cv.ORB_create()
    kp, des = orb.detectAndCompute(image, None)
    image = cv.drawKeypoints(image, kp, None)
    return image

def read_image(image, label, type):
    image = cv.imread(image)

    if image is None:
        print("Failed to open the image.")
    
    if type == 'harris':
        image = corner_harris(image)
        cv.imshow(label, image)
        cv.waitKey(0)
    
    elif type == 'surf':
        image = surf(image)
        cv.imshow(label, image)
        cv.waitKey(0)

ship1 = 'images/ship1.pgm'
ship2 = 'images/ship2.pgm'

read_image(ship1, 'ship1', 'harris')
read_image(ship2, 'ship2', 'harris')

castle1 = 'images/castle1.jpg'
castle2 = 'images/castle2.jpg'

read_image(castle1, 'castle1', 'harris')
read_image(castle2, 'castle2', 'harris') 

read_image(ship1, 'ship1', 'surf')
read_image(ship2, 'ship2', 'surf')
read_image(castle1, 'castle1', 'surf')
read_image(castle2, 'castle2', 'surf')