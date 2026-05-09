import cv2
import torch
from ultralytics import YOLO
from torchvision import transforms
from team_model import TeamCNN
from collections import Counter

# Configuración
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_yolo = YOLO("yolov8n.pt")
model_cnn = TeamCNN().to(device)
model_cnn.load_state_dict(torch.load("models/team_cnn.pth", map_location=device))
model_cnn.eval()

# Es CRUCIAL que esta transformación sea idéntica a la de cnn.py
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

cap = cv2.VideoCapture("videos/test1.mp4")

track_votes = {} 
final_teams = {} 

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    results = model_yolo.track(frame, persist=True)

    if results[0].boxes is not None and results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        ids = results[0].boxes.id.cpu().numpy().astype(int)

        for box, track_id in zip(boxes, ids):
            x1, y1, x2, y2 = map(int, box)
            
            # Ajuste de coherencia: Intentamos extraer el torso para la CNN
            h, w = y2 - y1, x2 - x1
            # Recortamos un poco los márgenes para centrarnos en la camiseta (coherencia con auto_label)
            player_img = frame[y1 + int(h*0.1):y1 + int(h*0.8), x1:x2]
            
            if player_img.size == 0: continue

            if track_id not in final_teams:
                input_tensor = transform(player_img).unsqueeze(0).to(device)
                with torch.no_grad():
                    pred = torch.argmax(model_cnn(input_tensor), dim=1).item()
                
                if track_id not in track_votes:
                    track_votes[track_id] = []
                
                track_votes[track_id].append(pred)

                # Votación: requerimos consistencia antes de asignar equipo permanentemente
                if len(track_votes[track_id]) > 15:
                    most_common = Counter(track_votes[track_id]).most_common(1)[0][0]
                    final_teams[track_id] = "Team A" if most_common == 0 else "Team B"
            
            team_name = final_teams.get(track_id, "Analizando...")
            color = (255, 0, 0) if team_name == "Team A" else (0, 0, 255)
            if team_name == "Analizando...": color = (0, 255, 255)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"ID:{track_id} {team_name}", (x1, y1-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    cv2.imshow("Análisis de Equipos Pro", frame)
    if cv2.waitKey(1) & 0xFF == 27: break

cap.release()
cv2.destroyAllWindows()