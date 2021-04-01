from pathlib import Path

import numpy as np
import tensorflow as tf
from moviepy.editor import *

# Adjust to use a different model/video file
MODEL_NAME = "fortnite.h5"
VIDEO_NAME = "example_fortnite_video.mp4"

ROOT_DIR = Path(__file__).parent.absolute().parent
VIDEO_PATH = ROOT_DIR.joinpath(f"assets/videos/{VIDEO_NAME}")
MODEL_PATH = ROOT_DIR.joinpath(f"models/{MODEL_NAME}")

try:
    model = tf.keras.models.load_model(MODEL_PATH)
except IOError:
    raise FileNotFoundError(f"No Model found at {MODEL_PATH}. You can download a test model from https://github.com/ContentAutomation/NeuralNetworks/releases/latest")

# VideoFileClip can only handle string as path (not PosixPath objects)
try:
    clip = VideoFileClip(str(VIDEO_PATH))
except IOError:
    raise FileNotFoundError(f"No Video found at {VIDEO_PATH}. You can download a test video from https://github.com/ContentAutomation/NeuralNetworks/releases/latest")

frames = []
# Check one frame each second except for the first one (e.g. 9 checks for a 10s video)
for t in range(1, int(clip.duration)):
    frame = clip.get_frame(t)
    resized_frame = tf.keras.preprocessing.image.smart_resize(frame, (224, 224), interpolation="nearest")
    frames.append(resized_frame)

# predictions[0] = game, predictions[1] = nogame
predictions = model.predict(np.array(frames))
percentage = np.average(predictions, axis=0)
print(f"Probability that video shows an ingame scene: {round(percentage[0] * 100, 4)}%")
clip.close()
