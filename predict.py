import tensorflow as tf
import numpy as np
from pathlib import Path
from moviepy.editor import *

GAME_NAME = "fortnite"
model_file_path = Path(
    f"/Users/christiancoenen/Google Drive/Social Media Automation/models/game_detection_{GAME_NAME}.h5"
)
model = tf.keras.models.load_model(model_file_path)
video_file_path = (
    "/Users/christiancoenen/Google Drive/Social Media Automation/" "Falsch predicted/predicted as Fortnite/clip_9.mp4"
)

clip = VideoFileClip(video_file_path)
frames = []
for t in range(1, int(clip.duration)):
    frame = clip.get_frame(t)
    resized_frame = tf.keras.preprocessing.image.smart_resize(frame, (224, 224), interpolation="nearest")
    frames.append(resized_frame)

# predictions[0] = game, predictions[1] = nogame
predictions = model.predict(np.array(frames))
percentage = np.average(predictions, axis=0)
print(f"Percentage Game:{percentage[0]} | Percentage No Game: {percentage[1]}")
clip.close()
