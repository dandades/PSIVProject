from ultralytics import YOLO
import torch
import cv2
import os
from torchvision import transforms

from team_model import TeamCNN

# CONFIG =======================
VIDEO_PATH = "videos/test1.mp4"
MODEL_PATH = "models/team_cnn.pth"
YOLO_PATH = "yolov8n.pt"

# ==============================

# TRANSFORM (igual que en training)
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

# Cargamos los modelos
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# CNN (clasificador equipos)
team_model = TeamCNN().to(device)
team_model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
team_model.eval()

# YOLO (detección + tracking)
yolo = YOLO(YOLO_PATH)

# VIDEO
cap = cv2.VideoCapture(VIDEO_PATH)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = yolo.track(frame, persist=True)

    for r in results:
        boxes = r.boxes

        for box in boxes:
            cls = int(box.cls[0])

            if cls != 0:
                continue  # solo personas

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            player_crop = frame[y1:y2, x1:x2]

            if player_crop.size == 0:
                continue

            # PREPROCESS + PREDICT
            img = transform(player_crop).unsqueeze(0).to(device)

            with torch.no_grad():
                output = team_model(img)
                pred = torch.argmax(output, dim=1).item()

            # Clasificamos por color
            # NOTE: No solo deberíamos hacer la media,
            if pred == 0:
                color = (255, 0, 0)  # Azul (Equipo A)
                label = "Team A"
            else:
                color = (0, 0, 255)  # Rojo (Equipo B)
                label = "Team B"

            # DIBUJAR
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.imshow("Result", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()