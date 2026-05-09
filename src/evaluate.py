import torch
import time
import os
import cv2
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torchvision import transforms
from PIL import Image
from team_model import TeamCNN

# ==========================================
# 1. MÉTRICAS DE LA CNN (Accuracy, Precision, Recall, F1)
# ==========================================
def evaluate_cnn(model_path, test_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TeamCNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    y_true = []
    y_pred = []
    inference_times = []

    print("--- Evaluando CNN ---")
    for class_label in ['0', '1']:
        class_path = os.path.join(test_dir, class_label)
        if not os.path.exists(class_path): continue

        for img_name in os.listdir(class_path):
            img_path = os.path.join(class_path, img_name)
            img = Image.open(img_path).convert('RGB')
            img_tensor = transform(img).unsqueeze(0).to(device)

            start_time = time.time()
            with torch.no_grad():
                output = model(img_tensor)
                pred = torch.argmax(output, dim=1).item()
            inference_times.append(time.time() - start_time)

            y_true.append(int(class_label))
            y_pred.append(pred)

    print(f"Accuracy:  {accuracy_score(y_true, y_pred):.4f}")
    print(f"Precision: {precision_score(y_true, y_pred):.4f}")
    print(f"Recall:    {recall_score(y_true, y_pred):.4f}")
    print(f"F1-Score:  {f1_score(y_true, y_pred):.4f}")
    print(f"Avg Inference Time: {np.mean(inference_times)*1000:.2f} ms")

# ==========================================
# 2. MÉTRICAS DE VIDEO (FPS, ID Switches manuales)
# ==========================================
def evaluate_video_performance(video_path):
    from ultralytics import YOLO
    model_yolo = YOLO("yolov8n.pt")
    cap = cv2.VideoCapture(video_path)
    
    frame_count = 0
    start_time = time.time()
    
    # Para ID Switches, en este nivel de proyecto se suele hacer una 
    # inspección visual de un clip corto y se anota manualmente.
    # Aquí calcularemos los FPS reales del sistema completo.

    print("\n--- Evaluando Rendimiento de Vídeo (FPS) ---")
    while cap.isOpened() and frame_count < 100: # Analizamos 100 frames para el test
        ret, frame = cap.read()
        if not ret: break
        
        # Simulación del pipeline completo (YOLO + Tracking)
        results = model_yolo.track(frame, persist=True, verbose=False)
        frame_count += 1
        
    end_time = time.time()
    total_time = end_time - start_time
    fps = frame_count / total_time
    
    print(f"FPS Reales del sistema: {fps:.2f}")
    print(f"Nota: ID Switches y MOTA requieren un archivo de anotación (.txt) Ground Truth.")

if __name__ == "__main__":
    # Ajusta estas rutas a tu proyecto
    evaluate_cnn("models/team_cnn.pth", "dataset/test")
    evaluate_video_performance("videos/test1.mp4")