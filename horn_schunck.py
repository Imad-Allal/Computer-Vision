import cv2
import numpy as np

# Charger les images
frames = []
for i in range(7, 15):
    frame_path = f'./Jardin/frame{i:02d}.png'
    frame = cv2.imread(frame_path, cv2.IMREAD_GRAYSCALE)
    frames.append(frame)

# Define region of interest (ROI) for the ball
roi = (10, 500, 100, 250)  # Example values, adjust as needed

# Détecter la balle avec mouvement vertical en utilisant la méthode Horn-Schunck
for i in range(len(frames) - 1):
    print(f'Loading frame {i}')

    # Calculate optical flow using Horn-Schunck method
    flow = cv2.calcOpticalFlowFarneback(frames[i], frames[i + 1], None, 0.5, 3, 15, 3, 5, 1.2, 0)

    # Extract horizontal and vertical components
    u = flow[:, :, 0]
    v = flow[:, :, 1]

    # Create a mask for the region of interest
    mask = np.zeros_like(frames[i + 1])

    # Draw the optical flow on the mask
    for y in range(roi[0], min(roi[1], frames[i].shape[0])):
        for x in range(roi[2], min(roi[3], frames[i].shape[1])):
            if abs(v[y, x]) > 0.1:  # Adjust the threshold as needed
                mask = cv2.line(mask, (x, y), (int(x + u[y, x]), int(y + v[y, x])), 255, 2)

    # Apply the mask to the frame
    result_frame = cv2.cvtColor(frames[i + 1], cv2.COLOR_GRAY2BGR)
    result_frame[roi[0]:min(roi[1], frames[i].shape[0]), roi[2]:min(roi[3], frames[i].shape[1])][mask[roi[0]:min(roi[1], frames[i].shape[0]), roi[2]:min(roi[3], frames[i].shape[1])] == 255] = [0, 0, 255]

    cv2.imshow('Optical Flow Detection (Horn-Schunck)', result_frame)
    cv2.waitKey(0)

cv2.destroyAllWindows()
