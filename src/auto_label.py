import os
import cv2
import numpy as np
from sklearn.cluster import KMeans
import shutil

input_dir = "dataset/raw_players"
output_dir = "dataset/team_classifier_auto"

os.makedirs(output_dir, exist_ok=True)

features = []
image_paths = []

for root, dirs, files in os.walk(input_dir):
    for file in files:
        path = os.path.join(root, file)

        img = cv2.imread(path)
        if img is None:
            continue

        h, w = img.shape[:2]
        area = h * w
        aspect_ratio = w / (h + 1e-6)

        # FILTROS (clave)
        if area < 2500:
            continue
        if aspect_ratio < 0.25 or aspect_ratio > 0.8:
            continue

        img_small = cv2.resize(img, (50, 50))

        mean_color = img_small.mean(axis=(0, 1))

        features.append(mean_color)
        image_paths.append(path)

features = np.array(features)

kmeans = KMeans(n_clusters=2, random_state=42)
labels = kmeans.fit_predict(features)

for i in range(2):
    os.makedirs(os.path.join(output_dir, str(i)), exist_ok=True)

for path, label in zip(image_paths, labels):
    filename = os.path.basename(path)
    dest = os.path.join(output_dir, str(label), filename)
    shutil.copy(path, dest)

print("Auto-labeling improved done!")