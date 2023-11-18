import cv2
import numpy as np

cap = cv2.VideoCapture('jardin.avi')
ret, old_frame = cap.read()
previous_frame = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if previous_frame is not None:
        optical_flow = cv2.calcOpticalFlowFarneback(previous_frame, frame_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

        # Utiliser les composantes X et Y du flux optique comme caractéristiques pour chaque pixel
        features = optical_flow.reshape((-1, 2))

        # Appliquer l'algorithme K-means
        k = 2  # Nombre de clusters, ajustez selon vos besoins
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
        _, labels, centers = cv2.kmeans(np.float32(features), k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

        # Remettre les étiquettes dans la forme de l'image
        labels = labels.reshape(optical_flow.shape[:2])

        # Assigner des couleurs différentes à chaque cluster pour la visualisation
        segmented_frame = np.zeros_like(frame)
        for i in range(k):
            segmented_frame[labels == i] = np.random.randint(0, 255, size=3)

        cv2.imshow('Segmented Frame', segmented_frame)

    previous_frame = frame_gray

    if cv2.waitKey(10) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
