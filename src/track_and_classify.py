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

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

cap = cv2.VideoCapture("videos/110.mp4")

# MEMORIA TEMPORAL
track_votes = {} # {id: [0, 1, 0, 0...]}
final_teams = {} # {id: "Team A"}

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    results = model_yolo.track(frame, persist=True)

    if results[0].boxes is not None and results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        ids = results[0].boxes.id.cpu().numpy().astype(int)

        for box, track_id in zip(boxes, ids):
            x1, y1, x2, y2 = map(int, box)
            player_img = frame[max(0,y1):y2, max(0,x1):x2]
            
            if player_img.size == 0: continue

            # Si el equipo no está decidido aún, acumulamos votos
            if track_id not in final_teams:
                input_tensor = transform(player_img).unsqueeze(0).to(device)
                with torch.no_grad():
                    pred = torch.argmax(model_cnn(input_tensor), dim=1).item()
                
                if track_id not in track_votes:
                    track_votes[track_id] = []
                
                track_votes[track_id].append(pred)

                # Tras 20 frames de observación, fijamos el equipo
                if len(track_votes[track_id]) > 20:
                    most_common = Counter(track_votes[track_id]).most_common(1)[0][0]
                    final_teams[track_id] = "Team A" if most_common == 0 else "Team B"
            
            # Dibujamos usando la decisión final (o la tendencia actual si aún está en proceso)
            team_name = final_teams.get(track_id, "Calculando...")
            color = (255, 0, 0) if team_name == "Team A" else (0, 0, 255)
            if team_name == "Calculando...": color = (0, 255, 255)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"ID:{track_id} {team_name}", (x1, y1-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    cv2.imshow("Fútbol Analysis", frame)
    if cv2.waitKey(1) & 0xFF == 27: break

cap.release()
cv2.destroyAllWindows()