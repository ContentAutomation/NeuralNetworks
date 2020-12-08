from langdetect import detect
from google.cloud import speech_v1p1beta1 as speech
import os

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "YouTube-94fecf3c77d1.json"

client = speech.SpeechClient()

speech_file = "audios/002_pt.wav"
first_lang = "en"
second_lang = ["es", "de", "pt"]

with open(speech_file, "rb") as audio_file:
    content = audio_file.read()

audio = speech.RecognitionAudio(content=content)

config = speech.RecognitionConfig(
    audio_channel_count=2,
    language_code=first_lang,
    alternative_language_codes=second_lang,
)

print("Waiting for operation to complete...")
response = client.recognize(config=config, audio=audio)

for i, result in enumerate(response.results):
    alternative = result.alternatives[0]
    lang = detect(alternative.transcript)
    print("-" * 20)
    print(f"First alternative of result {i}: {alternative}")
    print(f"Transcript: {alternative.transcript}")
    print(f"Detected language {lang}")
