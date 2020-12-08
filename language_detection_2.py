import speech_recognition as sr
from langdetect import detect

# es (spanish) basically filters out all languages like portuguese, italian, ...
CHECK_LANGUAGES = ["es", "de"]
FILEPATH = "audios/"
FILENAME = "005_en.wav"

r = sr.Recognizer()
audio_file = sr.AudioFile(FILEPATH + FILENAME)
with audio_file as source:
    audio = r.record(source)

# We start with assuming that the clip is english until proven otherwise
is_english = True
for language in CHECK_LANGUAGES:
    try:
        text = r.recognize_google(audio, language=language)
        print(f"Detected text: {text}")
        lang = detect(text)
        print(f"Detected language: '{lang}'")
        if lang != 'en' and len(text.split()) > 2:
            # If the check for a different language detects more than 2 words that are not classified as english by
            # the detect method, we assume that the clip is indeed not in english
            is_english = False
            break
    except sr.UnknownValueError:
        # That's good because no words from the unwanted language is detected, so we continue with the next one
        pass

print("Clip is in english!" if is_english else "Clips is not in english!")
