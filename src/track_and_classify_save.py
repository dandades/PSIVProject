import argparse
import cv2
import os
import torch
from collections import Counter

from torchvision import transforms
from ultralytics import YOLO

from team_model import TeamCNN


def parse_args():
    parser = argparse.ArgumentParser(
        description="Detecta jugadors, classifica equips i desa el video anotat."
    )
    parser.add_argument(
        "--video",
        default="videos/test1.mp4",
        help="Ruta del video d'entrada.",
    )
    parser.add_argument(
        "--output-dir",
        default="runs",
        help="Carpeta on es desara el video resultant.",
    )
    parser.add_argument(
        "--tracker",
        default="cfg/football_botsort.yaml",
        help="Configuracio del tracker d'Ultralytics.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.35,
        help="Llindar de confianca de YOLO. Valors mes alts redueixen deteccions inestables.",
    )
    parser.add_argument(
        "--stable-max-age",
        type=int,
        default=45,
        help="Frames que un ID estable pot quedar-se sense veure abans de descartar-lo.",
    )
    parser.add_argument(
        "--stable-iou",
        type=float,
        default=0.2,
        help="IoU minim per reconnectar un ID nou de YOLO amb un ID estable recent.",
    )
    parser.add_argument(
        "--stable-distance",
        type=float,
        default=90.0,
        help="Distancia maxima entre centres per reconnectar IDs estables.",
    )
    return parser.parse_args()


args = parse_args()

# Configuracio
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_yolo = YOLO("yolov8n.pt")
model_cnn = TeamCNN().to(device)
model_cnn.load_state_dict(torch.load("models/team_cnn.pth", map_location=device))
model_cnn.eval()

# Ha de ser identica a la transformacio de cnn.py
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

video_path = args.video
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    raise RuntimeError(f"No s'ha pogut obrir el video: {video_path}")

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

if fps <= 0:
    fps = 25

os.makedirs(args.output_dir, exist_ok=True)
video_name = os.path.splitext(os.path.basename(video_path))[0]
output_path = os.path.join(args.output_dir, f"tc_{video_name}.mp4")

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

if not out.isOpened():
    cap.release()
    raise RuntimeError(f"No s'ha pogut crear el video de sortida: {output_path}")

track_votes = {}
final_teams = {}
raw_to_stable_id = {}
stable_tracks = {}
next_stable_id = 1

print(f"Desant execucio a: {output_path}")


def box_iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter_area
    return inter_area / union if union > 0 else 0.0


def center_distance(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    acx = (ax1 + ax2) / 2
    acy = (ay1 + ay2) / 2
    bcx = (bx1 + bx2) / 2
    bcy = (by1 + by2) / 2
    return ((acx - bcx) ** 2 + (acy - bcy) ** 2) ** 0.5


def get_stable_id(raw_id, box, frame_index, active_stable_ids):
    global next_stable_id

    if raw_id in raw_to_stable_id:
        stable_id = raw_to_stable_id[raw_id]
    else:
        stable_id = None
        best_score = -1

        for candidate_id, track in stable_tracks.items():
            if candidate_id in active_stable_ids:
                continue

            age = frame_index - track["last_seen"]
            if age > args.stable_max_age:
                continue

            iou = box_iou(box, track["box"])
            distance = center_distance(box, track["box"])
            if iou < args.stable_iou and distance > args.stable_distance:
                continue

            score = iou - (distance / max(width, height))
            if score > best_score:
                best_score = score
                stable_id = candidate_id

        if stable_id is None:
            stable_id = next_stable_id
            next_stable_id += 1

        raw_to_stable_id[raw_id] = stable_id

    stable_tracks[stable_id] = {
        "box": box,
        "last_seen": frame_index,
        "raw_id": raw_id,
    }
    active_stable_ids.add(stable_id)
    return stable_id


frame_index = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model_yolo.track(
        frame,
        persist=True,
        tracker=args.tracker,
        conf=args.conf,
        classes=[0],
        verbose=False,
    )
    active_stable_ids = set()

    if results[0].boxes is not None and results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        ids = results[0].boxes.id.cpu().numpy().astype(int)

        for box, raw_track_id in zip(boxes, ids):
            x1, y1, x2, y2 = map(int, box)
            stable_id = get_stable_id(
                int(raw_track_id),
                (x1, y1, x2, y2),
                frame_index,
                active_stable_ids,
            )

            h = y2 - y1
            player_img = frame[y1 + int(h * 0.1):y1 + int(h * 0.8), x1:x2]

            if player_img.size == 0:
                continue

            if stable_id not in final_teams:
                input_tensor = transform(player_img).unsqueeze(0).to(device)
                with torch.no_grad():
                    pred = torch.argmax(model_cnn(input_tensor), dim=1).item()

                track_votes.setdefault(stable_id, []).append(pred)

                if len(track_votes[stable_id]) > 15:
                    most_common = Counter(track_votes[stable_id]).most_common(1)[0][0]
                    final_teams[stable_id] = "Team A" if most_common == 0 else "Team B"

            team_name = final_teams.get(stable_id, "Analitzant...")
            color = (255, 0, 0) if team_name == "Team A" else (0, 0, 255)
            if team_name == "Analitzant...":
                color = (0, 255, 255)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                frame,
                f"ID:{stable_id} {team_name}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
            )

    out.write(frame)

    cv2.imshow("Analisi de equips", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

    frame_index += 1

cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Video desat a: {output_path}")
