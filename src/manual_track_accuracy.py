import argparse
import csv
import json
import os
import subprocess
import sys
from collections import Counter

import cv2
import torch
from torchvision import transforms
from ultralytics import YOLO

from team_model import TeamCNN


TEAM_NAMES = {
    0: "Team A",
    1: "Team B",
}

def build_transform():
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def parse_args():
    parser = argparse.ArgumentParser(
        description="Genera evidencias por track_id y calcula accuracy con etiquetas manuales."
    )
    parser.add_argument("--video", default="videos/test1.mp4", help="Ruta del video a evaluar.")
    parser.add_argument("--yolo", default="yolov8n.pt", help="Ruta del modelo YOLO.")
    parser.add_argument("--model", default="models/team_cnn.pth", help="Ruta del modelo CNN entrenado.")
    parser.add_argument(
        "--output",
        default=None,
        help="Carpeta donde guardar evidencias. Por defecto usa manual_eval/tracks_[video-id].",
    )
    parser.add_argument(
        "--min-votes",
        type=int,
        default=16,
        help="Numero de predicciones por ID antes de fijar el equipo.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Limite opcional de frames a procesar para pruebas rapidas.",
    )
    return parser.parse_args()


def get_video_id(video_path):
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    return "".join(c if c not in '<>:"/\\|?*' else "_" for c in video_name)


def ensure_output_dir(output_dir, video_path):
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        return output_dir

    output_dir = os.path.join("manual_eval", f"tracks_{get_video_id(video_path)}")
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def clamp_box(box, frame_shape):
    height, width = frame_shape[:2]
    x1, y1, x2, y2 = map(int, box)
    x1 = max(0, min(x1, width - 1))
    x2 = max(0, min(x2, width))
    y1 = max(0, min(y1, height - 1))
    y2 = max(0, min(y2, height))
    return x1, y1, x2, y2


def crop_torso(frame, box):
    x1, y1, x2, y2 = clamp_box(box, frame.shape)
    height = y2 - y1
    torso_y1 = y1 + int(height * 0.1)
    torso_y2 = y1 + int(height * 0.8)
    return frame[torso_y1:torso_y2, x1:x2]


def predict_team(model, transform, device, player_img):
    input_tensor = transform(player_img).unsqueeze(0).to(device)
    with torch.no_grad():
        return torch.argmax(model(input_tensor), dim=1).item()


def save_evidence(output_dir, track_id, frame_id, frame, box, predicted_label):
    x1, y1, x2, y2 = clamp_box(box, frame.shape)
    team_name = TEAM_NAMES[predicted_label]
    player_crop = frame[y1:y2, x1:x2]
    safe_team_name = team_name.replace(" ", "_")
    base_name = f"id_{track_id:03d}_frame_{frame_id:06d}_pred_{safe_team_name}"
    crop_path = os.path.join(output_dir, f"{base_name}_crop.jpg")

    if player_crop.size > 0:
        cv2.imwrite(crop_path, player_crop)
    else:
        crop_path = ""

    return crop_path


def open_image(image_path):
    if not image_path:
        return

    absolute_path = os.path.abspath(image_path)
    try:
        if os.name == "nt":
            os.startfile(absolute_path)
        elif sys.platform == "darwin":
            subprocess.Popen(["open", absolute_path])
        else:
            subprocess.Popen(["xdg-open", absolute_path])
    except OSError as exc:
        print(f"No se ha podido abrir la imagen automaticamente: {exc}")


def collect_track_evidence(args, output_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo: {device}")
    print("Cargando modelos...")

    yolo = YOLO(args.yolo)
    team_model = TeamCNN().to(device)
    team_model.load_state_dict(torch.load(args.model, map_location=device))
    team_model.eval()
    transform = build_transform()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"No se ha podido abrir el video: {args.video}")

    track_votes = {}
    final_predictions = {}
    evidence = {}
    frame_id = 0

    print("Procesando video y generando una evidencia por ID...")
    while cap.isOpened():
        if args.max_frames is not None and frame_id >= args.max_frames:
            break

        ret, frame = cap.read()
        if not ret:
            break

        results = yolo.track(frame, persist=True, verbose=False)
        if not results or results[0].boxes is None or results[0].boxes.id is None:
            frame_id += 1
            continue

        boxes = results[0].boxes.xyxy.cpu().numpy()
        ids = results[0].boxes.id.cpu().numpy().astype(int)
        classes = results[0].boxes.cls.cpu().numpy().astype(int)

        for box, track_id, cls in zip(boxes, ids, classes):
            track_id = int(track_id)
            cls = int(cls)
            if cls != 0:
                continue

            player_img = crop_torso(frame, box)
            if player_img.size == 0:
                continue

            if track_id not in final_predictions:
                pred = predict_team(team_model, transform, device, player_img)
                track_votes.setdefault(track_id, []).append(pred)

                if len(track_votes[track_id]) >= args.min_votes:
                    predicted_label = int(Counter(track_votes[track_id]).most_common(1)[0][0])
                    final_predictions[track_id] = predicted_label
                else:
                    continue

            predicted_label = final_predictions[track_id]
            if track_id not in evidence:
                crop_path = save_evidence(
                    output_dir, track_id, frame_id, frame, box, predicted_label
                )
                evidence[track_id] = {
                    "track_id": int(track_id),
                    "frame_id": int(frame_id),
                    "prediction": int(predicted_label),
                    "prediction_name": TEAM_NAMES[predicted_label],
                    "votes": {
                        int(label): int(count)
                        for label, count in Counter(track_votes.get(track_id, [])).items()
                    },
                    "crop_path": crop_path,
                    "manual_answer": "",
                    "is_valid_player": "",
                    "team_correct": "",
                }

        if frame_id % 100 == 0:
            print(f"Frame {frame_id}: {len(evidence)} IDs con evidencia guardada")

        frame_id += 1

    cap.release()
    return evidence, track_votes, frame_id


def ask_manual_labels(evidence):
    print("\nRevision manual de cada ID")
    print("Opciones: s: Si, n: No, t: no corresponde a un jugador, k: saltar, q: salir")

    for track_id in sorted(evidence):
        item = evidence[track_id]
        print("\n" + "-" * 60)
        print(f"ID: {track_id}")
        print(f"Prediccion del modelo: {item['prediction_name']}")
        if item["crop_path"]:
            print(f"Recorte guardado: {item['crop_path']}")
            open_image(item["crop_path"])

        while True:
            raw_value = input("El modelo ha acertado el equipo? [s/n/t/k/q]: ").strip().lower()
            if raw_value == "q":
                return
            if raw_value in ("k", "skip", "saltar", ""):
                item["manual_answer"] = "skip"
                item["is_valid_player"] = ""
                item["team_correct"] = ""
                break
            if raw_value in ("s", "si", "y", "yes"):
                item["manual_answer"] = "s"
                item["is_valid_player"] = True
                item["team_correct"] = True
                break
            if raw_value in ("n", "no"):
                item["manual_answer"] = "n"
                item["is_valid_player"] = True
                item["team_correct"] = False
                break
            if raw_value in ("t", "otro", "none", "ninguno"):
                item["manual_answer"] = "t"
                item["is_valid_player"] = False
                item["team_correct"] = ""
                break

            print("Entrada no valida. Usa s, n, t, k o q.")


def write_results(output_dir, evidence):
    rows = []
    for track_id in sorted(evidence):
        row = evidence[track_id].copy()
        row["track_id"] = int(row["track_id"])
        row["frame_id"] = int(row["frame_id"])
        row["prediction"] = int(row["prediction"])
        row["votes"] = {
            str(int(label)): int(count)
            for label, count in row["votes"].items()
        }
        rows.append(row)

    csv_path = os.path.join(output_dir, "manual_track_accuracy.csv")
    json_path = os.path.join(output_dir, "manual_track_accuracy.json")

    fieldnames = [
        "track_id",
        "frame_id",
        "prediction",
        "prediction_name",
        "manual_answer",
        "is_valid_player",
        "team_correct",
        "votes",
        "crop_path",
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            csv_row = row.copy()
            csv_row["votes"] = json.dumps(csv_row["votes"], sort_keys=True)
            writer.writerow(csv_row)

    with open(json_path, "w", encoding="utf-8") as json_file:
        json.dump(rows, json_file, indent=2)

    return csv_path, json_path


def print_metrics(evidence, total_frames, csv_path, json_path):
    reviewed = [
        item
        for item in evidence.values()
        if item["manual_answer"] not in ("", "skip")
    ]
    valid_players = [
        item
        for item in reviewed
        if item["is_valid_player"] is True
    ]

    valid_count = len(valid_players)
    team_correct = sum(1 for item in valid_players if item["team_correct"] is True)

    valid_player_precision = valid_count / len(reviewed) if reviewed else 0.0
    team_accuracy = team_correct / valid_count if valid_count else 0.0

    non_player_count = sum(1 for item in reviewed if item["is_valid_player"] is False)
    skipped_count = sum(1 for item in evidence.values() if item["manual_answer"] == "skip")

    print("\nResultados")
    print(f"Frames procesados: {total_frames}")
    print(f"IDs con evidencia: {len(evidence)}")
    print(f"IDs revisados: {len(reviewed)}")
    print(f"IDs saltados: {skipped_count}")
    print(f"IDs marcados como no jugador: {non_player_count}")
    print(
        "Precision de IDs validos como jugadores: "
        f"{valid_count}/{len(reviewed)} = {valid_player_precision:.4f}"
    )
    print(
        "Precision del equipo entre jugadores validos: "
        f"{team_correct}/{valid_count} = {team_accuracy:.4f}"
    )
    print(f"CSV: {csv_path}")
    print(f"JSON: {json_path}")


def main():
    args = parse_args()
    output_dir = ensure_output_dir(args.output, args.video)

    evidence, track_votes, total_frames = collect_track_evidence(args, output_dir)

    if not evidence:
        print("No se ha generado evidencia. Prueba con un video mas largo o baja --min-votes.")
        return

    votes_path = os.path.join(output_dir, "all_track_votes.json")
    with open(votes_path, "w", encoding="utf-8") as votes_file:
        json.dump({str(k): v for k, v in sorted(track_votes.items())}, votes_file, indent=2)

    print(f"\nEvidencias guardadas en: {output_dir}")
    print(f"Numero de IDs a procesar manualmente: {len(evidence)}")
    print("Se abrira automaticamente el recorte correspondiente en cada iteracion.")
    input("Pulsa Enter cuando estes listo para revisar los IDs...")

    ask_manual_labels(evidence)
    csv_path, json_path = write_results(output_dir, evidence)
    print_metrics(evidence, total_frames, csv_path, json_path)


if __name__ == "__main__":
    main()
