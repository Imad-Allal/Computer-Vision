import cv2
import numpy as np

# Charger les images
frames = []
for i in range(7, 15):
    frame_path = f'./Jardin/frame{i:02d}.png'
    frame = cv2.imread(frame_path, cv2.IMREAD_GRAYSCALE)
    frames.append(frame)

# Define region of interest (ROI) for the ball
roi = (0, 500, 100, 250)  # Example values, adjust as needed

# Détecter la balle avec mouvement vertical en utilisant la fonction Lucas-Kanade
for i in range(len(frames) - 1):
    print(f'Loading frame {i}')

    # Calculate optical flow using Lucas-Kanade method
    lk_params = dict(winSize=(15, 15),
                     maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.3))
    
    p0 = cv2.goodFeaturesToTrack(frames[i], mask=None, **dict(maxCorners=100, qualityLevel=0.3, minDistance=7))
    
    p1, st, err = cv2.calcOpticalFlowPyrLK(frames[i], frames[i + 1], p0, None, **lk_params)

    # Select good points
    good_new = p1[st == 1]
    good_old = p0[st == 1]

    # Create a mask for the region of interest
    mask = np.zeros_like(frames[i + 1])

    # Draw the tracks on the mask
    for new, old in zip(good_new, good_old):
        a, b = new.ravel()
        c, d = old.ravel()
        mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), 255, 2)

    # Apply the mask to the frame
    result_frame = cv2.cvtColor(frames[i + 1], cv2.COLOR_GRAY2BGR)
    result_frame[roi[0]:roi[1], roi[2]:roi[3]][mask[roi[0]:roi[1], roi[2]:roi[3]] == 255] = [0, 0, 255]

    cv2.imshow('Optical Flow Detection', result_frame)
    cv2.waitKey(0)

cv2.destroyAllWindows()
