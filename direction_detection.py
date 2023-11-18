import cv2
import numpy as np

cap = cv2.VideoCapture('robot.avi')
ret, old_frame = cap.read()
previous_frame = None
hsv = np.zeros_like(old_frame)
hsv[..., 1] = 255

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


    if previous_frame is not None:
        optical_flow = cv2.calcOpticalFlowFarneback(previous_frame, frame_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

        flow_to_right = optical_flow[..., 0] > 1
        flow_to_left = optical_flow[..., 0] < -1

        num_pixels_right = np.sum(flow_to_right)
        num_pixels_left = np.sum(flow_to_left)

        if num_pixels_right > num_pixels_left:
            msg = 'To Left'
        else:
            msg = 'To Right'

        cv2.imshow('Optical Flow', frame)
        cv2.putText(frame, msg, (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.imshow('Optical Flow', frame)

        # Encoding: convert the algorithm's output into Polar coordinates
        mag, ang = cv2.cartToPolar(optical_flow[..., 0], optical_flow[..., 1])
        # Use Hue and Value to encode the Optical Flow
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    
        # Convert HSV image into BGR for demo
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        cv2.imshow("optical flow", bgr)

    previous_frame = frame_gray

    if cv2.waitKey(10) & 0xFF == 27: 
        break

cap.release()
cv2.destroyAllWindows()
