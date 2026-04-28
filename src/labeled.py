import cv2
import os
import shutil

input_dir = "dataset/raw_players"
out_a = "dataset/team_classifier/A"
out_b = "dataset/team_classifier/B"

os.makedirs(out_a, exist_ok=True)
os.makedirs(out_b, exist_ok=True)

images = os.listdir(input_dir) # Lists all images in the directory

for i in images:
    path = os.path.join(input_dir, i)
    img = cv2.imread(path)

    cv2.imshow("Player", img)

    key = cv2.waitKey(0) # indefinite wait

    # Classify as A
    if key == ord('a'):
        shutil.copy(path, os.path.join(out_a, i))

    # Classify as B
    elif key == ord('b'):
        shutil.copy(path, os.path.join(out_b, i))

    elif key == 27: # ESC = 27 ASCII
        break

# Closes al OpenCV windows
cv2.destroyAllWindows()