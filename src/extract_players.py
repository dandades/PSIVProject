from ultralytics import YOLO
import cv2
import os

model = YOLO("yolov8n.pt")

videos_dir = "videos"
output_dir = "dataset/raw_players"
os.makedirs(output_dir, exist_ok=True)

frame_global_id = 0  # contador global

for video_name in os.listdir(videos_dir):
    video_path = os.path.join(videos_dir, video_name)

    # Solo vídeos (por si se ha colado algo que no es un vídeo en el diectorio)
    if not video_path.endswith((".mp4", ".avi", ".mov")):
        continue

    print(f"Processing: {video_name}")

    # Creamos una carpeta para cada vídeo
    video_folder_name = os.path.splitext(video_name)[0]  # quita .mp4
    video_output_dir = os.path.join(output_dir, video_folder_name)
    os.makedirs(video_output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    frame_id = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.track(frame, persist=True)

        for r in results:
            boxes = r.boxes

            for i, box in enumerate(boxes):
                cls = int(box.cls[0])

                if cls == 0:  # persona
                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    cut = frame[y1:y2, x1:x2]

                    if cut.size == 0:
                        continue

                    # Guardar dentro de la carpeta del vídeo
                    save_path = os.path.join(
                        video_output_dir,
                        f"frame{frame_id}_id{i}.jpg"
                    )

                    cv2.imwrite(save_path, cut)

        frame_id += 1
        frame_global_id += 1

    cap.release()

print("Done extracting from all videos")