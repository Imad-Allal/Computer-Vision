import cv2
import numpy as np

def create_panorama(I1, I2, points_img1, points_img2):
    if len(points_img1) < 4 or len(points_img2) < 4:
        print("Insufficient points to create a panorama. Please select more points.")
        return None

    matchedPoints1 = np.array(points_img1)
    matchedPoints2 = np.array(points_img2)

    M, _ = cv2.findHomography(matchedPoints2, matchedPoints1, cv2.RANSAC, 5.0)

    h1, w1 = I1.shape[:2]
    h2, w2 = I2.shape[:2]

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

case1_images = []
for i in range(1, 6):
    image_path = f'images/case1/{i}.JPG'
    image = cv2.imread(image_path)
    # image = cv2.resize(image, None, fx=0.25, fy=0.25)
    case1_images.append(image)

panoramas = []

all_points_img1 = []
all_points_img2 = []

def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(all_points_img1) < len(case1_images) and len(all_points_img2) < len(case1_images):
            all_points_img1.append((x, y))
            print(f"Selected point in image {len(all_points_img1)}: ({x}, {y})")
        elif len(all_points_img1) < len(case1_images):
            print("Please select points in image 2.")
        elif len(all_points_img2) < len(case1_images):
            all_points_img2.append((x, y))
            print(f"Selected point in image {len(all_points_img2)}: ({x}, {y})")
            if len(all_points_img2) == len(case1_images):
                print("Points selection complete for all images.")

cv2.namedWindow('Image')
cv2.setMouseCallback('Image', mouse_callback)

for i in range(len(case1_images)):
    I1 = case1_images[i]
    points_img1 = []
    points_img2 = []
    
    # Show the image for manual point selection
    cv2.imshow('Image', I1)
    cv2.waitKey(0)
    
    # Store the selected points for image i
    all_points_img1.append(points_img1)
    all_points_img2.append(points_img2)

    # Close the window after selection
    cv2.destroyWindow('Image')

for i in range(len(case1_images) - 1):
    I1 = case1_images[i]
    I2 = case1_images[i + 1]
    points_img1 = all_points_img1[i]
    points_img2 = all_points_img2[i]
    
    transformed_image = create_panorama(I1, I2, points_img1, points_img2)
    panoramas.append(transformed_image)

cv2.imshow('Final Panorama', panoramas[-1])
cv2.waitKey(0)
cv2.destroyAllWindows()
