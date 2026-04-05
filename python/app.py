import logging
from http import HTTPStatus
from itertools import chain

from flask import Flask, Response, jsonify, request, stream_with_context
from werkzeug.utils import secure_filename

from audio_utils import (
    AudioProcessingError,
    VoiceChunkArtifacts,
    pcm_duration_seconds,
    preprocess_reference_audio,
    safe_unlink,
    save_upload_to_temp,
    silence_pcm_bytes,
    validate_wav_file,
    validate_audio_filename,
    wav_stream_header,
)
from config import LOG_LEVEL, MAX_TEXT_LENGTH, TEMP_DIR
from rvc_service import RVCService
from tts_service import TTSService


logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
LOGGER = logging.getLogger("voice_cloner_api")

app = Flask(__name__)
tts_service = TTSService.get_instance()
rvc_service = RVCService.get_instance()


def error_response(message: str, status_code: int) -> Response:
    return jsonify({"status": "error", "message": message}), status_code


def validate_request_payload():
    text = (request.form.get("text") or "").strip()
    LOGGER.info("Input received. text_length=%s files=%s", len(text), list(request.files.keys()))
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
    LOGGER.info("Input validation passed. audio_filename=%s", audio_file.filename)
    return text, audio_file, None


@app.post("/generate")
def generate():
    temp_input_path = None
    cleaned_reference_path = None
    first_artifacts: VoiceChunkArtifacts | None = None
    use_rvc = False

    try:
        text, audio_file, validation_error = validate_request_payload()
        if validation_error:
            return validation_error

        safe_name = secure_filename(audio_file.filename)
        temp_input_path = save_upload_to_temp(audio_file.stream, safe_name, TEMP_DIR)
        LOGGER.info("Upload stored temporarily at %s", temp_input_path)
        trim_silence = (request.form.get("trim_silence", "true").lower() != "false")
        cleaned_reference_path = preprocess_reference_audio(
            input_path=temp_input_path,
            output_dir=TEMP_DIR,
            trim_silence=trim_silence,
        )
        cleaned_metadata = validate_wav_file(cleaned_reference_path, "Preprocessed reference WAV")
        LOGGER.info("Preprocessed reference audio is valid: %s", cleaned_metadata)

        tts_settings = tts_service.build_settings(request.form)
        rvc_settings = rvc_service.build_settings(request.form)
        use_rvc = rvc_service.is_ready()
        sample_rate = rvc_service.output_sample_rate if use_rvc else tts_service.sample_rate
        chunk_stream = tts_service.synthesize_chunks(text, cleaned_reference_path, tts_settings)
        LOGGER.info("Pipeline selected. use_rvc=%s sample_rate=%s", use_rvc, sample_rate)

        successful_chunks = []
        chunk_error_messages = []
        for chunk_index, tts_chunk in chunk_stream:
            try:
                if use_rvc:
                    LOGGER.info("Processing chunk_index=%s through RVC. xtts_bytes=%s", chunk_index, len(tts_chunk))
                    artifacts = rvc_service.convert_chunk(
                        chunk_index=chunk_index,
                        pcm_audio=tts_chunk,
                        input_sample_rate=tts_service.sample_rate,
                        reference_wav=cleaned_reference_path,
                        settings=rvc_settings,
                    )
                    if artifacts.pcm_bytes is None or len(artifacts.pcm_bytes) == 0:
                        raise RuntimeError(f"Audio generation failed: RVC returned empty PCM for chunk_index={chunk_index}")
                else:
                    if tts_chunk is None or len(tts_chunk) == 0:
                        raise RuntimeError(f"Audio generation failed: XTTS returned empty PCM for chunk_index={chunk_index}")
                    artifacts = VoiceChunkArtifacts(
                        pcm_bytes=tts_chunk,
                        input_wav_path="",
                        output_wav_path="",
                        resampled_output_path="",
                    )
                LOGGER.info(
                    "Prepared streaming chunk_index=%s duration=%.3f bytes=%s stage=%s",
                    chunk_index,
                    pcm_duration_seconds(artifacts.pcm_bytes, sample_rate),
                    len(artifacts.pcm_bytes),
                    "rvc" if use_rvc else "xtts",
                )
                first_artifacts = artifacts
                successful_chunks.append((chunk_index, artifacts))
                break
            except Exception as exc:
                LOGGER.exception("Chunk failed before stream start. chunk_index=%s", chunk_index)
                chunk_error_messages.append(f"chunk {chunk_index}: {exc}")
                continue

        if not successful_chunks or first_artifacts is None:
            safe_unlink(temp_input_path, cleaned_reference_path)
            return error_response(
                "Audio generation failed. No non-empty audio chunk was produced. " + "; ".join(chunk_error_messages[:3]),
                HTTPStatus.INTERNAL_SERVER_ERROR,
            )

        def remaining_successful_chunks():
            for chunk_index, tts_chunk in chunk_stream:
                try:
                    if use_rvc:
                        LOGGER.info("Processing streaming chunk_index=%s through RVC. xtts_bytes=%s", chunk_index, len(tts_chunk))
                        artifacts = rvc_service.convert_chunk(
                            chunk_index=chunk_index,
                            pcm_audio=tts_chunk,
                            input_sample_rate=tts_service.sample_rate,
                            reference_wav=cleaned_reference_path,
                            settings=rvc_settings,
                        )
                        if artifacts.pcm_bytes is None or len(artifacts.pcm_bytes) == 0:
                            raise RuntimeError(f"Audio generation failed: empty PCM for chunk_index={chunk_index}")
                    else:
                        if tts_chunk is None or len(tts_chunk) == 0:
                            raise RuntimeError(f"Audio generation failed: XTTS returned empty PCM for chunk_index={chunk_index}")
                        artifacts = VoiceChunkArtifacts(
                            pcm_bytes=tts_chunk,
                            input_wav_path="",
                            output_wav_path="",
                            resampled_output_path="",
                        )
                    yield chunk_index, artifacts
                except Exception:
                    LOGGER.exception("Chunk failed during active stream; continuing. chunk_index=%s", chunk_index)
                    continue

        def generate_audio():
            artifacts: VoiceChunkArtifacts | None = first_artifacts
            first_chunk = True
            emitted_payload = False
            try:
                header = wav_stream_header(sample_rate)
                LOGGER.info("Streaming started. WAV header bytes=%s sample_rate=%s", len(header), sample_rate)
                yield header
                for chunk_index, artifacts in chain(successful_chunks, remaining_successful_chunks()):
                    if artifacts.pcm_bytes is None or len(artifacts.pcm_bytes) == 0:
                        raise RuntimeError(f"Audio generation failed: empty streaming PCM for chunk_index={chunk_index}")
                    if not first_chunk:
                        spacer = silence_pcm_bytes(sample_rate)
                        LOGGER.info("Yielding silence spacer. bytes=%s", len(spacer))
                        yield spacer
                    yield artifacts.pcm_bytes
                    emitted_payload = True
                    LOGGER.info(
                        "Yielding audio chunk. chunk_index=%s chunk_size=%s duration=%.3f stage=%s",
                        chunk_index,
                        len(artifacts.pcm_bytes),
                        pcm_duration_seconds(artifacts.pcm_bytes, sample_rate),
                        "rvc" if use_rvc else "xtts",
                    )
                    safe_unlink(*artifacts.cleanup_paths())
                    artifacts = None
                    first_chunk = False
            finally:
                LOGGER.info("Streaming finished. emitted_payload=%s", emitted_payload)
                if artifacts is not None:
                    safe_unlink(*artifacts.cleanup_paths())
                safe_unlink(temp_input_path, cleaned_reference_path)

        headers = {
            "Content-Type": "audio/wav",
            "Cache-Control": "no-store",
            "X-Audio-Sample-Rate": str(sample_rate),
            "X-Voice-Pipeline": "xtts-rvc" if use_rvc else "xtts-only",
            "Transfer-Encoding": "chunked",
        }
        return Response(stream_with_context(generate_audio()), mimetype="audio/wav", headers=headers)

    except AudioProcessingError as exc:
        LOGGER.exception("Audio preprocessing failed")
        if first_artifacts is not None:
            safe_unlink(*first_artifacts.cleanup_paths())
        safe_unlink(temp_input_path, cleaned_reference_path)
        return error_response(str(exc), HTTPStatus.BAD_REQUEST)
    except ValueError as exc:
        LOGGER.exception("Request validation or synthesis failed")
        if first_artifacts is not None:
            safe_unlink(*first_artifacts.cleanup_paths())
        safe_unlink(temp_input_path, cleaned_reference_path)
        return error_response(str(exc), HTTPStatus.BAD_REQUEST)
    except RuntimeError as exc:
        LOGGER.exception("Runtime error in /generate")
        if first_artifacts is not None:
            safe_unlink(*first_artifacts.cleanup_paths())
        safe_unlink(temp_input_path, cleaned_reference_path)
        return error_response(str(exc), HTTPStatus.SERVICE_UNAVAILABLE)
    except Exception as exc:
        LOGGER.exception("Unhandled failure in /generate")
        if first_artifacts is not None:
            safe_unlink(*first_artifacts.cleanup_paths())
        safe_unlink(temp_input_path, cleaned_reference_path)
        return error_response(f"Voice cloning failed: {exc}", HTTPStatus.INTERNAL_SERVER_ERROR)


@app.post("/clone")
def clone_alias():
    return generate()


@app.route("/health", methods=["GET", "POST"])
def health():
    try:
        return jsonify(
            {
                "status": "ok",
                "pipeline": "xtts-rvc" if rvc_service.is_ready() else "xtts-only",
                "tts": tts_service.health(),
                "rvc": rvc_service.health(),
            }
        ), HTTPStatus.OK
    except Exception as exc:
        LOGGER.exception("Health check failed")
        return jsonify({"status": "error", "message": str(exc)}), HTTPStatus.INTERNAL_SERVER_ERROR


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, threaded=True)
