import os
import cv2
import numpy as np
from sklearn.cluster import KMeans
import shutil

input_dir = "dataset/raw_players"
output_dir = "dataset/team_classifier_auto"

os.makedirs(output_dir, exist_ok=True)

# Leer imágenes y extraer color medio
features = []
image_paths = []

for file in os.listdir(input_dir):
    path = os.path.join(input_dir, file)
    img = cv2.imread(path)

    if img is None:
        continue

    # Redimensionar para acelerar
    img_small = cv2.resize(img, (50, 50))

    # Media de color (RGB)
    mean_color = img_small.mean(axis=(0, 1))
    features.append(mean_color)
    image_paths.append(path)

features = np.array(features)

# KMeans → 2 clusters (2 equipos)
kmeans = KMeans(n_clusters=2, random_state=42)
labels = kmeans.fit_predict(features)

# Crear carpetas
for i in range(2):
    os.makedirs(os.path.join(output_dir, str(i)), exist_ok=True)

# Guardar imágenes en clusters
for path, label in zip(image_paths, labels):
    filename = os.path.basename(path)
    dest = os.path.join(output_dir, str(label), filename)
    shutil.copy(path, dest)

print("Auto-labeling done!")