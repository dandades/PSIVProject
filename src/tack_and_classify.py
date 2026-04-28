import cv2
from ultralytics import YOLO
from team_classifier import classify_team

model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture("videos/test1.mp4")

team_memory = {}

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model.track(frame, persist=True)

    if results[0].boxes is not None:
        for box in results[0].boxes:

            cls = int(box.cls[0])
            if cls != 0:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            track_id = int(box.id[0]) if box.id is not None else None

            player_img = frame[y1:y2, x1:x2]
            if player_img.size == 0:
                continue

            if track_id not in team_memory:
                team_memory[track_id] = classify_team(player_img)

            team = team_memory[track_id]

            color = (255, 0, 0) if team == "Team A" else (0, 0, 255)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{team} ID:{track_id}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    cv2.imshow("Result", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()