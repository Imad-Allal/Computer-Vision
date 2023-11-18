import cv2
import numpy as np

# Charger la séquence vidéo
video_path = 'Ball.avi'  # Mettez le chemin correct vers votre vidéo
cap = cv2.VideoCapture(video_path)

# Lire la première image pour initialiser le flux optique
ret, frame1 = cap.read()
prev_gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

# Initialiser le détecteur d'objets en mouvement
fgbg = cv2.createBackgroundSubtractorMOG2()

while True:
    # Lire la trame suivante
    ret, frame2 = cap.read()
    if not ret:
        break

    # Convertir l'image en niveaux de gris
    gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # Calculer le flux optique avec la méthode Farneback
    flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    # Calculer la magnitude et la direction du flux optique
    magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    # Appliquer une seuil pour détecter les régions en mouvement
    threshold = 10  # Ajustez le seuil selon vos besoins
    motion_mask = magnitude > threshold

    # Appliquer le masque sur l'image originale pour mettre en évidence les objets en mouvement
    result_frame = frame2.copy()
    result_frame[motion_mask] = [0, 0, 255]  # Couleur rouge pour les objets en mouvement

    # Afficher les trames
    cv2.imshow('Motion Segmentation', result_frame)

    # Mettre à jour l'image précédente
    prev_gray = gray

    # Sortir de la boucle si la touche 'q' est enfoncée
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

# Libérer les ressources
cap.release()
cv2.destroyAllWindows()
