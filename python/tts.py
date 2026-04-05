from flask import Flask, request, send_file
from TTS.api import TTS
import uuid
import torch

app = Flask(__name__)

tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2")
# tts = TTS(model_name="tts_models/de/thorsten/tacotron2-DDC")

# Move to CPU/GPU automatically
device = "cuda" if torch.cuda.is_available() else "cpu"
tts.to(device)

from utils.voice_convert import convert_to_wav
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@app.route("/generate", methods=["POST"])
def generate():
    data = request.json
    text = data.get("text")

    input_voice = os.path.join(BASE_DIR, "../voices/harsh-bsdk.mp3")
    clean_voice = os.path.join(BASE_DIR, "../voices/harsh-bsdk.wav")
    # 🔥 convert first
    convert_to_wav(input_voice, clean_voice)

    output_path = f"../outputs/{uuid.uuid4()}.wav"

    tts.tts_to_file(
        text=text,
        speaker_wav=clean_voice,
        language="en",
        file_path=output_path,
        split_sentences=True,
        temperature=0.6,
        length_penalty=1.0,
        repetition_penalty=2.0,
    )

    return send_file(output_path, as_attachment=True)
if __name__ == "__main__":
    app.run(port=5001)