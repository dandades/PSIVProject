from ultralytics import YOLO
import cv2
import os

#NOTE: We are using yolo8n.pt AI model to detect the players and the ball on each frame
model = YOLO("yolov8n.pt") 

video_path = "videos/test1.mp4"
cap = cv2.VideoCapture(video_path) # opens the video for reading it frame per frame

output_dir = "dataset/raw_players"
os.makedirs(output_dir, exist_ok=True)

frame_id = 0 # frames counter

# We iterate all the image frames
while True:
    continues, frame = cap.read() # continues is a boolean value. True if a frame exists | False if the video has ended
    if not continues: 
        break
    
    # Executes AI YOLO model and applies tracking
    results = model.track(frame, persist=True)

    for r in results:
        # bounding boxes extraction
        boxes = r.boxes

        for i, box in enumerate(boxes):
            cls = int(box.cls[0]) # Object class extraction (Since we are using COCO, 0 = person)

            # ONLY PERSONS
            if cls == 0:
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                cut  = frame[y1:y2, x1:x2] # Bounding Box content

                if cut.size == 0:
                    continue 

                save_path = f"{output_dir}/frame{frame_id}_id{i}.jpg"
                cv2.imwrite(save_path, cut)

    frame_id += 1

# Closes the video without errors
cap.release()
print("Done extracting players")