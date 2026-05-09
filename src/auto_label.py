import os
import cv2
import numpy as np
from sklearn.cluster import KMeans
import shutil

input_dir = "dataset/raw_players"
output_dir = "dataset/team_classifier_auto"
CONFIDENCE_THRESHOLD = 0.2  # Margen de seguridad para descartar dudas

os.makedirs(output_dir, exist_ok=True)

features = []
image_paths = []

for root, dirs, files in os.walk(input_dir):
    for file in files:
        path = os.path.join(root, file)
        img = cv2.imread(path)
        if img is None: continue

        h, w = img.shape[:2]
        aspect_ratio = w / (h + 1e-6)

        # FILTROS DE FORMA (Solo rectángulos verticales)
        if h < 50 or aspect_ratio < 0.25 or aspect_ratio > 0.65:
            continue

        # MÁSCARA DE TORSO: Ignoramos cabeza y pies (donde suele haber piel o césped)
        torso = img[int(h*0.2):int(h*0.6), int(w*0.2):int(w*0.8)]
        if torso.size == 0: continue

        # CONVERSIÓN A HSV (Robustez ante sombras)
        hsv = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV)
        
        # Filtro opcional: Solo colores que NO sean verde (césped)
        lower_green = np.array([35, 40, 40])
        upper_green = np.array([85, 255, 255])
        mask = cv2.inRange(hsv, lower_green, upper_green)
        non_green_mask = cv2.bitwise_not(mask)
        
        # Media de color solo de la parte que no es verde
        mean_val = cv2.mean(hsv, mask=non_green_mask)[:3]
        features.append(mean_val)
        image_paths.append(path)

features = np.array(features)
kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
labels = kmeans.fit_predict(features)

# Cálculo de distancias para el umbral de confianza
distances = kmeans.transform(features)

for i in range(2):
    os.makedirs(os.path.join(output_dir, str(i)), exist_ok=True)

for idx, (path, label) in enumerate(zip(image_paths, labels)):
    # Si la distancia entre clusters es muy pequeña, es una "duda" y no lo usamos para entrenar
    d0, d1 = distances[idx][0], distances[idx][1]
    rel_dist = abs(d0 - d1) / (max(d0, d1) + 1e-6)
    
    if rel_dist < CONFIDENCE_THRESHOLD:
        continue # Descartar imágenes ambiguas

    filename = os.path.basename(path)
    dest = os.path.join(output_dir, str(label), filename)
    shutil.copy(path, dest)

print(f"Auto-labeling finalizado. Imágenes procesadas: {len(image_paths)}")