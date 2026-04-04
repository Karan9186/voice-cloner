from flask import Flask, request, send_file
from TTS.api import TTS
import uuid

app = Flask(__name__)

# Load model once (important for performance)
tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2")

@app.route("/generate", methods=["POST"])
def generate():
    data = request.json
    text = data.get("text")

    output_path = f"../outputs/{uuid.uuid4()}.wav"

    tts.tts_to_file(
        text=text,
        speaker_wav="../voices/myvoice.wav",
        language="en",
        file_path=output_path
    )

    return send_file(output_path, as_attachment=True)

if __name__ == "__main__":
    app.run(port=5001)