import cv2
import numpy as np

# Read images
I1 = cv2.imread('images/campus_01.jpg')
I2 = cv2.imread('images/campus_02.jpg')

# Resize imaged to half of their original size
I1 = cv2.resize(I1, None, fx=0.25, fy=0.25)
I2 = cv2.resize(I2, None, fx=0.25, fy=0.25)


# Get the original height and width of the image
I_height, I_width = I1.shape[:2]

# Convert images to grayscale
gI1 = cv2.cvtColor(I1, cv2.COLOR_BGR2GRAY)
gI2 = cv2.cvtColor(I2, cv2.COLOR_BGR2GRAY)

# Visualize image
# cv2.imshow('Color Image', cv2.resize(gI1, (I_width//5, I_height//5)))
#cv2.waitKey(0)

# Create ORB detector and descriptor
orb = cv2.ORB_create()

# Detect and compute descriptors for the first image
P1, F1 = orb.detectAndCompute(gI1, None)
image1 = cv2.drawKeypoints(gI1, P1, None)
# Detect and compute descriptors for the second image
P2, F2 = orb.detectAndCompute(gI2, None)
image2 = cv2.drawKeypoints(gI2, P2, None)

points_img1 = []
points_img2 = []

def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(points_img1) < 4 and len(points_img2) < 4:
            points_img1.append((x, y))
            print(f"Selected point in image 1: ({x}, {y})")
        elif len(points_img1) < 4:
            print("Please select points in image 2.")
        elif len(points_img2) < 4:
            points_img2.append((x, y))
            print(f"Selected point in image 2: ({x}, {y})")
        else:
            print("Points selection complete.")


cv2.imshow('Image 1', I1)
cv2.setMouseCallback('Image 1', mouse_callback)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imshow('Image 2', I2)
cv2.setMouseCallback('Image 2', mouse_callback)
cv2.waitKey(0)
cv2.destroyAllWindows()

print("Selected points in image 1:", points_img1)
print("Selected points in image 2:", points_img2)

def create_panorama(I1, I2, points_img1, points_img2):

    if len(points_img1) < 4 or len(points_img2) < 4:
        print("Insufficient points to create a panorama. Please select more points.")
    else:
        matchedPoints1 = np.array(points_img1)
        matchedPoints2 = np.array(points_img2)

        M, _ = cv2.findHomography(matchedPoints2, matchedPoints1, cv2.RANSAC, 5.0)

        h1, w1 = gI1.shape
        h2, w2 = gI2.shape

        corners1 = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)
        corners2 = np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)

        corners2_transformed = cv2.perspectiveTransform(corners2, M)
        corners = np.concatenate((corners1, corners2_transformed), axis=0)

        x_min, y_min = np.int32(corners.min(axis=0).ravel())
        x_max, y_max = np.int32(corners.max(axis=0).ravel())

        transformed_offset = (-x_min, -y_min)
        transformed_image = cv2.warpPerspective(I2, M, (x_max - x_min, y_max - y_min))
        transformed_image[transformed_offset[1]:h1 + transformed_offset[1], transformed_offset[0]:w1 + transformed_offset[0]] = I1

        return transformed_image
    
transformed_image_campus = create_panorama(I1, I2, points_img1, points_img2)
cv2.imshow('Panorama', transformed_image_campus)
cv2.waitKey(0)
cv2.destroyAllWindows()


# Read images
I1 = cv2.imread('images/campus_01.jpg')
I2 = cv2.imread('images/campus_02.jpg')

# Resize imaged to half of their original size
I1 = cv2.resize(I1, None, fx=0.25, fy=0.25)
I2 = cv2.resize(I2, None, fx=0.25, fy=0.25)

points_img1 = []
points_img2 = []

transformed_image_case1 = create_panorama(I1, I2, points_img1, points_img2)







