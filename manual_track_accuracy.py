import argparse
import csv
import json
import os
import subprocess
import sys
import time

import cv2

from extract_players import extract_player_tracks
from labeling import label_player_teams
from utils import read_video


TEAM_NAMES = {
    0: "Team A",
    1: "Team B",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Genera evidencias por track_id y permite evaluar manualmente "
            "si cada ID es un jugador valido y si su equipo se clasifico bien."
        )
    )
    parser.add_argument(
        "--video",
        default="input_videos/110.mp4",
        help="Ruta del video a evaluar."
    )
    parser.add_argument(
        "--model",
        default="models/best.pt",
        help="Ruta del modelo YOLO."
    )
    parser.add_argument(
        "--tracks",
        default=None,
        help="Ruta opcional del stub de tracks. Si no se indica, se deriva del video."
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Carpeta donde guardar evidencias y resultados."
    )
    parser.add_argument(
        "--no-stubs",
        action="store_true",
        help="Recalcula los tracks en lugar de cargar/guardar el stub."
    )
    parser.add_argument(
        "--min-track-frames",
        type=int,
        default=5,
        help="Minimo de frames consecutivos para conservar un track."
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Limite opcional de frames para pruebas rapidas."
    )
    parser.add_argument(
        "--sample-every",
        type=int,
        default=10,
        help="Frecuencia de muestreo de frames para entrenar K-means."
    )
    parser.add_argument(
        "--collect-only",
        action="store_true",
        help="Solo genera evidencias y CSV/JSON, sin pedir revision manual."
    )
    parser.add_argument(
        "--no-open",
        action="store_true",
        help="No abre automaticamente las imagenes durante la revision manual."
    )
    return parser.parse_args()


def get_video_id(video_path):
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    return "".join(c if c not in '<>:"/\\|?*' else "_" for c in video_name)


def ensure_output_dir(output_dir, video_path):
    if output_dir is None:
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


def crop_player(frame, bbox):
    x1, y1, x2, y2 = clamp_box(bbox, frame.shape)
    if x2 <= x1 or y2 <= y1:
        return None
    return frame[y1:y2, x1:x2]


def save_evidence(output_dir, track_id, frame_num, frame, bbox, predicted_team):
    player_crop = crop_player(frame, bbox)
    if player_crop is None or player_crop.size == 0:
        return "", ""

    prediction_name = TEAM_NAMES[predicted_team].replace(" ", "_")
    base_name = f"id_{track_id:03d}_frame_{frame_num:06d}_pred_{prediction_name}"
    crop_path = os.path.join(output_dir, f"{base_name}_crop.jpg")
    frame_path = os.path.join(output_dir, f"{base_name}_frame.jpg")

    cv2.imwrite(crop_path, player_crop)

    context_frame = frame.copy()
    x1, y1, x2, y2 = clamp_box(bbox, frame.shape)
    cv2.rectangle(context_frame, (x1, y1), (x2, y2), (0, 255, 255), 3)
    cv2.putText(
        context_frame,
        f"ID {track_id} - {TEAM_NAMES[predicted_team]}",
        (x1, max(25, y1 - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2
    )
    cv2.imwrite(frame_path, context_frame)

    return crop_path, frame_path


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


def collect_evidence(args, output_dir):
    frames = read_video(args.video)
    if args.max_frames is not None:
        frames = frames[:args.max_frames]

    if not frames:
        raise RuntimeError(f"No se han podido leer frames del video: {args.video}")

    _, tracks, stub_path = extract_player_tracks(
        frames,
        input_video_path=args.video,
        model_path=args.model,
        use_stubs=not args.no_stubs,
        min_track_frames=args.min_track_frames,
        stub_path=args.tracks
    )
    label_player_teams(frames, tracks, sample_every=args.sample_every)

    evidence = {}
    for frame_num, player_track in enumerate(tracks["players"]):
        if frame_num >= len(frames):
            break

        frame = frames[frame_num]
        for track_id, track in player_track.items():
            predicted_team = int(track["team"])
            crop = crop_player(frame, track["bbox"])
            if crop is None or crop.size == 0:
                continue

            crop_area = int(crop.shape[0] * crop.shape[1])
            current_item = evidence.get(track_id)
            if current_item is not None and crop_area <= current_item["crop_area"]:
                continue
            if current_item is not None:
                for path_key in ("crop_path", "frame_path"):
                    previous_path = current_item.get(path_key)
                    if previous_path and os.path.exists(previous_path):
                        os.remove(previous_path)

            crop_path, frame_path = save_evidence(
                output_dir,
                int(track_id),
                int(frame_num),
                frame,
                track["bbox"],
                predicted_team
            )

            evidence[int(track_id)] = {
                "track_id": int(track_id),
                "frame_num": int(frame_num),
                "prediction": predicted_team,
                "prediction_name": TEAM_NAMES[predicted_team],
                "is_valid_player": "",
                "team_correct": "",
                "manual_answer": "",
                "crop_path": crop_path,
                "frame_path": frame_path,
                "crop_area": crop_area,
            }

        if frame_num % 100 == 0:
            print(f"Frame {frame_num}: {len(evidence)} IDs con evidencia")

    print(f"Stub de tracks usado: {stub_path}")
    return evidence, len(frames)


def ask_manual_labels(evidence, open_images=True):
    print("\nRevision manual de cada ID")
    print("Opciones:")
    print("  s: si, es jugador valido y el equipo es correcto")
    print("  n: no, es jugador valido pero el equipo es incorrecto")
    print("  t: no corresponde a un jugador valido")
    print("  k: saltar")
    print("  q: parar y guardar lo revisado hasta el momento\n")

    for track_id in sorted(evidence):
        item = evidence[track_id]
        print("-" * 60)
        print(f"ID: {track_id}")
        print(f"Frame evidencia: {item['frame_num']}")
        print(f"Prediccion: {item['prediction_name']}")
        print(f"Recorte: {item['crop_path']}")
        print(f"Frame:   {item['frame_path']}")

        if open_images:
            open_image(item["frame_path"] or item["crop_path"])

        while True:
            raw_value = input("Resultado manual [s/n/t/k/q]: ").strip().lower()
            if raw_value == "q":
                return
            if raw_value in ("k", "skip", "saltar", ""):
                item["manual_answer"] = "skip"
                item["is_valid_player"] = ""
                item["team_correct"] = ""
                break
            if raw_value in ("s", "si", "y", "yes"):
                item["manual_answer"] = "valid_correct_team"
                item["is_valid_player"] = True
                item["team_correct"] = True
                break
            if raw_value in ("n", "no"):
                item["manual_answer"] = "valid_wrong_team"
                item["is_valid_player"] = True
                item["team_correct"] = False
                break
            if raw_value in ("t", "invalid", "invalido", "no_jugador"):
                item["manual_answer"] = "invalid_player"
                item["is_valid_player"] = False
                item["team_correct"] = ""
                break

            print("Entrada no valida. Usa s, n, t, k o q.")


def write_results(output_dir, evidence):
    rows = []
    for track_id in sorted(evidence):
        row = evidence[track_id].copy()
        row.pop("crop_area", None)
        rows.append(row)

    csv_path = os.path.join(output_dir, "manual_track_accuracy.csv")
    json_path = os.path.join(output_dir, "manual_track_accuracy.json")

    fieldnames = [
        "track_id",
        "frame_num",
        "prediction",
        "prediction_name",
        "is_valid_player",
        "team_correct",
        "manual_answer",
        "crop_path",
        "frame_path",
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    with open(json_path, "w", encoding="utf-8") as json_file:
        json.dump(rows, json_file, indent=2)

    return csv_path, json_path


def print_metrics(evidence, total_frames, csv_path, json_path, execution_time):
    reviewed = [
        item for item in evidence.values()
        if item["manual_answer"] not in ("", "skip")
    ]
    valid_players = [
        item for item in reviewed
        if item["is_valid_player"] is True
    ]

    valid_count = len(valid_players)
    invalid_count = sum(1 for item in reviewed if item["is_valid_player"] is False)
    correct_team_count = sum(
        1 for item in valid_players
        if item["team_correct"] is True
    )
    skipped_count = sum(
        1 for item in evidence.values()
        if item["manual_answer"] == "skip"
    )

    valid_player_precision = valid_count / len(reviewed) if reviewed else 0.0
    team_accuracy = correct_team_count / valid_count if valid_count else 0.0

    print("\nResultados")
    print(f"Frames procesados: {total_frames}")
    print(f"Tiempo de ejecución: {execution_time:.2f} segundos")
    print(f"IDs con evidencia: {len(evidence)}")
    print(f"IDs revisados: {len(reviewed)}")
    print(f"IDs saltados: {skipped_count}")
    print(f"IDs marcados como no jugador valido: {invalid_count}")
    print(
        "Precision de IDs validos como jugadores: "
        f"{valid_count}/{len(reviewed)} = {valid_player_precision:.4f}"
    )
    print(
        "Accuracy de equipo entre jugadores validos: "
        f"{correct_team_count}/{valid_count} = {team_accuracy:.4f}"
    )
    print(f"CSV: {csv_path}")
    print(f"JSON: {json_path}")


def main():
    args = parse_args()
    output_dir = ensure_output_dir(args.output, args.video)

    start_time = time.time()
    evidence, total_frames = collect_evidence(args, output_dir)
    execution_time = time.time() - start_time

    if not evidence:
        print("No se ha generado evidencia. Prueba con otro video o sin stubs.")
        return

    print(f"\nEvidencias guardadas en: {output_dir}")
    print(f"Tiempo de ejecución del procesamiento automático: {execution_time:.2f} segundos")
    print(f"Numero de IDs a revisar: {len(evidence)}")

    if not args.collect_only:
        input("Pulsa Enter cuando estes listo para revisar manualmente...")
        ask_manual_labels(evidence, open_images=not args.no_open)

    csv_path, json_path = write_results(output_dir, evidence)
    print_metrics(evidence, total_frames, csv_path, json_path, execution_time)


if __name__ == "__main__":
    main()
