import logging
import os
import tempfile
import threading
from dataclasses import asdict, dataclass
from typing import Dict, Optional

from audio_utils import (
    AudioProcessingError,
    VoiceChunkArtifacts,
    ffmpeg_resample,
    pcm_duration_seconds,
    read_wav_pcm,
    run_subprocess,
    validate_wav_file,
    write_pcm_wav,
)
from config import (
    DEFAULT_OUTPUT_SAMPLE_RATE,
    RVC_F0_METHOD,
    RVC_FILTER_RADIUS,
    RVC_HOP_LENGTH,
    RVC_INDEX_PATH,
    RVC_INDEX_RATE,
    RVC_INFER_SCRIPT,
    RVC_MODEL_PATH,
    RVC_PITCH,
    RVC_PROTECT,
    RVC_PYTHON_BIN,
    TEMP_DIR,
)


LOGGER = logging.getLogger(__name__)


@dataclass
class RVCSettings:
    pitch: int = RVC_PITCH
    index_rate: float = RVC_INDEX_RATE
    protect: float = RVC_PROTECT
    filter_radius: int = RVC_FILTER_RADIUS
    hop_length: int = RVC_HOP_LENGTH
    f0_method: str = RVC_F0_METHOD


class RVCService:
    _instance: Optional["RVCService"] = None
    _instance_lock = threading.Lock()

    def __init__(self) -> None:
        self.model_path = RVC_MODEL_PATH
        self.index_path = RVC_INDEX_PATH
        self.infer_script = RVC_INFER_SCRIPT
        self.python_bin = RVC_PYTHON_BIN
        self.output_sample_rate = DEFAULT_OUTPUT_SAMPLE_RATE
        self._convert_lock = threading.Lock()

    @classmethod
    def get_instance(cls) -> "RVCService":
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def build_settings(self, payload: Dict[str, str]) -> RVCSettings:
        settings = RVCSettings()
        if payload.get("pitch"):
            settings.pitch = int(payload["pitch"])
        if payload.get("index_rate"):
            settings.index_rate = float(payload["index_rate"])
        if payload.get("protect"):
            settings.protect = float(payload["protect"])
        if payload.get("filter_radius"):
            settings.filter_radius = int(payload["filter_radius"])
        if payload.get("hop_length"):
            settings.hop_length = int(payload["hop_length"])
        if payload.get("f0_method"):
            settings.f0_method = payload["f0_method"]
        return settings

    def is_ready(self) -> bool:
        return bool(
            self.infer_script
            and os.path.exists(self.infer_script)
            and self.model_path
            and os.path.exists(self.model_path)
            and (not self.index_path or os.path.exists(self.index_path))
        )

    def convert_chunk(
        self,
        chunk_index: int,
        pcm_audio: bytes,
        input_sample_rate: int,
        reference_wav: str,
        settings: RVCSettings,
    ) -> VoiceChunkArtifacts:
        self._ensure_ready()
        os.makedirs(TEMP_DIR, exist_ok=True)
        LOGGER.info(
            "RVC conversion started. chunk_index=%s input_bytes=%s input_sample_rate=%s reference_wav=%s",
            chunk_index,
            len(pcm_audio) if pcm_audio is not None else None,
            input_sample_rate,
            reference_wav,
        )

        if pcm_audio is None or len(pcm_audio) == 0:
            raise RuntimeError(f"Audio generation failed: empty RVC input for chunk_index={chunk_index}")

        input_fd, input_wav_path = tempfile.mkstemp(prefix=f"tts_chunk_{chunk_index}_", suffix=".wav", dir=TEMP_DIR)
        os.close(input_fd)
        raw_fd, raw_output_path = tempfile.mkstemp(prefix=f"rvc_chunk_{chunk_index}_", suffix=".wav", dir=TEMP_DIR)
        os.close(raw_fd)
        final_fd, final_output_path = tempfile.mkstemp(prefix=f"rvc_24k_{chunk_index}_", suffix=".wav", dir=TEMP_DIR)
        os.close(final_fd)

        write_pcm_wav(input_wav_path, pcm_audio, input_sample_rate)
        input_metadata = validate_wav_file(input_wav_path, "RVC input WAV")
        LOGGER.info("RVC input validated: %s", input_metadata)

        command = [
            self.python_bin,
            self.infer_script,
            "--input_path",
            input_wav_path,
            "--output_path",
            raw_output_path,
            "--model_path",
            self.model_path,
            "--index_path",
            self.index_path,
            "--pitch",
            str(settings.pitch),
            "--index_rate",
            str(settings.index_rate),
            "--protect",
            str(settings.protect),
            "--filter_radius",
            str(settings.filter_radius),
            "--hop_length",
            str(settings.hop_length),
            "--f0_method",
            settings.f0_method,
            "--speaker_wav",
            reference_wav,
        ]

        try:
            with self._convert_lock:
                run_subprocess(command, "RVC conversion failed.")
            raw_metadata = validate_wav_file(raw_output_path, "RVC raw output WAV")
            LOGGER.info("RVC raw output shape/metadata: %s", raw_metadata)
            ffmpeg_resample(raw_output_path, final_output_path, self.output_sample_rate)
            final_metadata = validate_wav_file(final_output_path, "RVC resampled output WAV")
            LOGGER.info("RVC resampled output metadata: %s", final_metadata)
            output_sample_rate, pcm_bytes = read_wav_pcm(final_output_path)
            if pcm_bytes is None or len(pcm_bytes) == 0:
                raise RuntimeError(f"Audio generation failed: RVC output is empty for chunk_index={chunk_index}")
            LOGGER.info(
                "RVC output ready. chunk_index=%s duration=%.3f bytes=%s sample_rate=%s",
                chunk_index,
                pcm_duration_seconds(pcm_bytes, output_sample_rate),
                len(pcm_bytes),
                output_sample_rate,
            )
            return VoiceChunkArtifacts(
                pcm_bytes=pcm_bytes,
                input_wav_path=input_wav_path,
                output_wav_path=raw_output_path,
                resampled_output_path=final_output_path,
            )
        except (AudioProcessingError, RuntimeError):
            for path in (input_wav_path, raw_output_path, final_output_path):
                if os.path.exists(path):
                    os.unlink(path)
            raise
        except Exception as exc:
            LOGGER.exception("Unexpected RVC failure on chunk_index=%s", chunk_index)
            for path in (input_wav_path, raw_output_path, final_output_path):
                if os.path.exists(path):
                    os.unlink(path)
            raise RuntimeError(f"Audio generation failed during RVC conversion: {exc}") from exc

    def _ensure_ready(self) -> None:
        errors = []
        if not self.infer_script:
            errors.append("RVC_INFER_SCRIPT")
        elif not os.path.exists(self.infer_script):
            errors.append(f"RVC infer script not found: {self.infer_script}")

        if not self.model_path:
            errors.append("RVC_MODEL_PATH")
        elif not os.path.exists(self.model_path):
            errors.append(f"RVC model not found: {self.model_path}")

        if self.index_path and not os.path.exists(self.index_path):
            errors.append(f"RVC index not found: {self.index_path}")

        if errors:
            raise RuntimeError("RVC is not configured correctly. " + "; ".join(errors))
        LOGGER.info(
            "RVC configuration validated. infer_script=%s model_path=%s index_path=%s",
            self.infer_script,
            self.model_path,
            self.index_path,
        )

    def health(self) -> Dict[str, object]:
        return {
            "status": "ok",
            "configured": bool(self.infer_script and self.model_path),
            "ready": self.is_ready(),
            "infer_script": self.infer_script,
            "model_path": self.model_path,
            "index_path": self.index_path,
            "output_sample_rate": self.output_sample_rate,
            "defaults": asdict(RVCSettings()),
        }
