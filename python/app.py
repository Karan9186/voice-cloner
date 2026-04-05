import logging
import os
from http import HTTPStatus

from flask import Flask, Response, jsonify, request, stream_with_context
from werkzeug.utils import secure_filename

from audio_utils import (
    AudioProcessingError,
    preprocess_reference_audio,
    safe_unlink,
    save_upload_to_temp,
    validate_audio_filename,
    wav_stream_header,
)
from tts_service import TTSService


logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
LOGGER = logging.getLogger("xtts_flask")

MAX_TEXT_LENGTH = int(os.getenv("MAX_TEXT_LENGTH", "2500"))
TEMP_DIR = os.getenv("XTTS_TEMP_DIR") or os.path.join(os.path.dirname(__file__), "tmp")
os.makedirs(TEMP_DIR, exist_ok=True)

app = Flask(__name__)
tts_service = TTSService.get_instance()


def error_response(message: str, status_code: int) -> Response:
    return jsonify({"status": "error", "message": message}), status_code


def validate_request_payload():
    text = (request.form.get("text") or "").strip()
    if not text:
        return None, None, error_response("Field 'text' is required.", HTTPStatus.BAD_REQUEST)
    if len(text) > MAX_TEXT_LENGTH:
        return None, None, error_response(
            f"Input text exceeds MAX_TEXT_LENGTH={MAX_TEXT_LENGTH}. Submit shorter text for low-VRAM streaming.",
            HTTPStatus.BAD_REQUEST,
        )

    audio_file = request.files.get("audio")
    if audio_file is None or not audio_file.filename:
        return None, None, error_response("Multipart field 'audio' is required.", HTTPStatus.BAD_REQUEST)

    validate_audio_filename(audio_file.filename)
    return text, audio_file, None


@app.post("/clone")
def clone_voice():
    temp_input_path = None
    cleaned_reference_path = None

    try:
        text, audio_file, validation_error = validate_request_payload()
        if validation_error:
            return validation_error

        safe_name = secure_filename(audio_file.filename)
        _, extension = os.path.splitext(safe_name)
        temp_input_path = save_upload_to_temp(audio_file.stream, extension or ".wav")
        trim_silence = (request.form.get("trim_silence", "true").lower() != "false")
        cleaned_reference_path = preprocess_reference_audio(
            temp_input_path,
            output_dir=TEMP_DIR,
            trim_silence=trim_silence,
        )

        settings = tts_service.build_settings(request.form)
        sample_rate = tts_service.sample_rate
        audio_stream = tts_service.synthesize_stream(text, cleaned_reference_path, settings)

        def generate():
            try:
                yield wav_stream_header(sample_rate)
                for pcm_chunk in audio_stream:
                    yield pcm_chunk
            finally:
                safe_unlink(temp_input_path, cleaned_reference_path)

        headers = {
            "Content-Type": "audio/wav",
            "X-Audio-Sample-Rate": str(sample_rate),
            "Cache-Control": "no-store",
        }
        return Response(stream_with_context(generate()), headers=headers)

    except AudioProcessingError as exc:
        LOGGER.exception("Audio preprocessing failed")
        safe_unlink(temp_input_path, cleaned_reference_path)
        return error_response(str(exc), HTTPStatus.BAD_REQUEST)
    except ValueError as exc:
        LOGGER.exception("Request validation or synthesis failed")
        safe_unlink(temp_input_path, cleaned_reference_path)
        return error_response(str(exc), HTTPStatus.BAD_REQUEST)
    except Exception as exc:
        LOGGER.exception("Unhandled failure in /clone")
        safe_unlink(temp_input_path, cleaned_reference_path)
        return error_response(f"Voice cloning failed: {exc}", HTTPStatus.INTERNAL_SERVER_ERROR)


@app.route("/health", methods=["GET", "POST"])
def health():
    try:
        return jsonify(tts_service.health()), HTTPStatus.OK
    except Exception as exc:
        LOGGER.exception("Health check failed")
        return jsonify({"status": "error", "message": str(exc)}), HTTPStatus.INTERNAL_SERVER_ERROR


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5001")), threaded=True)
