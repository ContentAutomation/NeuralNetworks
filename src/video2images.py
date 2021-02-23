import cv2
import os
from pathlib import Path

ROOT_DIR = Path(__file__).parent.absolute().parent

# Adjust for different paths/configuration
VIDEO_PATH = ROOT_DIR.joinpath(f"assets/videos")
IMAGES_PATH = ROOT_DIR.joinpath(f"assets/images")
SKIPPED_FRAMES = 300
img_counter = 0
frames_counter = 0
directory_handle = os.fsencode(str(VIDEO_PATH))

IMAGES_PATH.mkdir(parents=True, exist_ok=True)
for file in os.listdir(directory_handle):
    filename = os.fsdecode(file)
    idx_counter = 0
    if filename.endswith(".mp4") or filename.endswith(".mpg"):
        frames_counter = 0
        capture = cv2.VideoCapture(f"{VIDEO_PATH}/{filename}")
        # Check how many frames the video consists of
        nr_frames = capture.get(7)
        while capture.isOpened():
            status, image = capture.read()
            if status and frames_counter <= nr_frames:
                cv2.imwrite(f"{IMAGES_PATH}/{filename}_0000{idx_counter}.jpg", image)
                print(f"Images: {img_counter}")
                idx_counter += 1
                img_counter += 1
                frames_counter += SKIPPED_FRAMES
                capture.set(1, frames_counter)
            else:
                capture.release()
cv2.destroyAllWindows()
