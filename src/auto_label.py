import os
import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import shutil

# CONFIGURACIÓN
input_dir = "dataset/raw_players"
output_dir = "dataset/team_classifier_auto"
CONFIDENCE_THRESHOLD = 0.25 # Margen de seguridad aumentado para máxima calidad

os.makedirs(output_dir, exist_ok=True)
os.makedirs(os.path.join(output_dir, "duda"), exist_ok=True)

def extract_features(img):
    h, w = img.shape[:2]
    # 1. FOCO EN EL TORSO (60% central para evitar ruido de fondo/botas)
    torso = img[int(h*0.2):int(h*0.7), int(w*0.15):int(w*0.85)]
    if torso.size == 0: return None

    # 2. MÁSCARA PARA IGNORAR CÉSPED (HSV)
    hsv = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV)
    lower_green = np.array([35, 40, 40])
    upper_green = np.array([85, 255, 255])
    green_mask = cv2.inRange(hsv, lower_green, upper_green)
    player_mask = cv2.bitwise_not(green_mask)

    # Si no hay suficiente "jugador" tras quitar el verde, descartamos
    if cv2.countNonZero(player_mask) < (torso.size / 6): return None

    # --- CARACTERÍSTICA A: COLOR (Media y Desviación) ---
    mean_color = cv2.mean(hsv, mask=player_mask)[:3]
    # Desviación para captar si la camiseta tiene varios colores
    std_color = cv2.meanStdDev(hsv, mask=player_mask)[1].flatten()

    # --- CARACTERÍSTICA B: TEXTURA (Filtro de Gabor) ---
    # Usamos un filtro de Gabor para detectar líneas/texturas
    gray_torso = cv2.cvtColor(torso, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getGaborKernel((21, 21), 5.0, np.pi/4, 10.0, 0.5, 0, ktype=cv2.CV_32F)
    fimg = cv2.filter2D(gray_torso, cv2.CV_8U, kernel)
    texture_score = fimg.mean()

    # --- CARACTERÍSTICA C: BORDES (Complejidad del diseño) ---
    edges = cv2.Canny(gray_torso, 100, 200)
    edge_density = cv2.countNonZero(edges) / (edges.size + 1e-6)

    # Combinamos todo en un vector de características
    # [H, S, V, stdH, stdS, stdV, Texture, Edges]
    vector = np.concatenate([mean_color, std_color, [texture_score, edge_density]])
    return vector

features = []
image_paths = []

print("Extrayendo características avanzadas (Color + Textura + Bordes)...")

for root, dirs, files in os.walk(input_dir):
    for file in files:
        path = os.path.join(root, file)
        img = cv2.imread(path)
        if img is None: continue
        
        # Filtro de forma básico
        h, w = img.shape[:2]
        if h < 50 or (w/h) > 0.8: continue

        feat_vector = extract_features(img)
        if feat_vector is not None:
            features.append(feat_vector)
            image_paths.append(path)

if not features:
    print("No se extrajeron características. Revisa las imágenes de entrada.")
    exit()

# NORMALIZACIÓN (Vital cuando mezclamos escalas: color vs textura)
features = np.array(features)
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# K-MEANS
kmeans = KMeans(n_clusters=2, random_state=42, n_init=15)
labels = kmeans.fit_predict(features_scaled)
distances = kmeans.transform(features_scaled)

# CLASIFICACIÓN CON FILTRO DE DUDA
print("Organizando dataset...")
for i in range(2): os.makedirs(os.path.join(output_dir, str(i)), exist_ok=True)

for idx, (path, label) in enumerate(zip(image_paths, labels)):
    d0, d1 = distances[idx][0], distances[idx][1]
    margin = abs(d0 - d1) / (max(d0, d1) + 1e-7)
    
    filename = os.path.basename(path)
    if margin < CONFIDENCE_THRESHOLD:
        dest = os.path.join(output_dir, "duda", filename)
    else:
        dest = os.path.join(output_dir, str(label), filename)
    
    shutil.copy(path, dest)

print(f"Hecho. Revisa {output_dir} para validar los clusters.")