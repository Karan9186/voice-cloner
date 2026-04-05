import logging
import os
import threading
from dataclasses import asdict, dataclass
from typing import Dict, Generator, Optional

import torch
from TTS.api import TTS

from audio_utils import chunk_text, pcm16le_bytes, pcm_duration_seconds, text_word_count, validate_wav_file
from config import (
    DEFAULT_OUTPUT_SAMPLE_RATE,
    XTTS_LANGUAGE,
    XTTS_LENGTH_PENALTY,
    XTTS_MODEL_NAME,
    XTTS_REPETITION_PENALTY,
    XTTS_SPEED,
    XTTS_TEMPERATURE,
    XTTS_TOP_P,
)


LOGGER = logging.getLogger(__name__)

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True


@dataclass
class TTSSettings:
    language: str = XTTS_LANGUAGE
    temperature: float = XTTS_TEMPERATURE
    top_p: float = XTTS_TOP_P
    speed: float = XTTS_SPEED
    length_penalty: float = XTTS_LENGTH_PENALTY
    repetition_penalty: float = XTTS_REPETITION_PENALTY


class TTSService:
    _instance: Optional["TTSService"] = None
    _instance_lock = threading.Lock()

    def __init__(self) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = XTTS_MODEL_NAME
        self._load_lock = threading.Lock()
        self._synthesis_lock = threading.Lock()
        self._tts: Optional[TTS] = None
        self._sample_rate = DEFAULT_OUTPUT_SAMPLE_RATE
        self._using_half = False

    @classmethod
    def get_instance(cls) -> "TTSService":
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    @property
    def sample_rate(self) -> int:
        self.ensure_model_loaded()
        return self._sample_rate

    def ensure_model_loaded(self) -> None:
        if self._tts is not None:
            return

        with self._load_lock:
            if self._tts is not None:
                return

            LOGGER.info("Loading XTTS model '%s' on %s", self.model_name, self.device)
            with torch.inference_mode():
                tts = TTS(model_name=self.model_name, progress_bar=False, gpu=self.device == "cuda")
                tts.to(self.device)

            if self.device == "cuda":
                self._enable_half_precision(tts)

            synthesizer = getattr(tts, "synthesizer", None)
            sample_rate = getattr(synthesizer, "output_sample_rate", None)
            if isinstance(sample_rate, int) and sample_rate > 0:
                self._sample_rate = sample_rate

            self._tts = tts
            LOGGER.info(
                "XTTS ready. device=%s sample_rate=%s half_precision=%s",
                self.device,
                self._sample_rate,
                self._using_half,
            )

    def _enable_half_precision(self, tts: TTS) -> None:
        try:
            tts_model = getattr(getattr(tts, "synthesizer", None), "tts_model", None)
            if tts_model is not None:
                tts_model.half()
                self._using_half = True
        except Exception as exc:
            LOGGER.warning("Half precision could not be enabled cleanly: %s", exc)

    def build_settings(self, payload: Dict[str, str]) -> TTSSettings:
        settings = TTSSettings()
        if payload.get("language"):
            settings.language = payload["language"]
        if payload.get("temperature"):
            settings.temperature = float(payload["temperature"])
        if payload.get("top_p"):
            settings.top_p = float(payload["top_p"])
        if payload.get("speed"):
            settings.speed = float(payload["speed"])
        if payload.get("length_penalty"):
            settings.length_penalty = float(payload["length_penalty"])
        if payload.get("repetition_penalty"):
            settings.repetition_penalty = float(payload["repetition_penalty"])
        return settings

    def synthesize_chunks(
        self,
        text: str,
        speaker_wav: str,
        settings: TTSSettings,
    ) -> Generator[tuple[int, bytes], None, None]:
        self.ensure_model_loaded()
        assert self._tts is not None
        self._validate_speaker_wav(speaker_wav)

        chunks = chunk_text(text)
        if not chunks:
            raise ValueError("No valid text chunks were produced from the supplied text.")

        LOGGER.info(
            "XTTS synthesis started. words=%s chunks=%s device=%s language=%s",
            text_word_count(text),
            len(chunks),
            self.device,
            settings.language,
        )

        for chunk in chunks:
            LOGGER.info(
                "XTTS generating chunk %s/%s text_len=%s",
                chunk.index + 1,
                len(chunks),
                len(chunk.text),
            )
            pcm_bytes = self._synthesize_chunk(chunk.text, speaker_wav, settings)
            LOGGER.info(
                "XTTS generated %.3f seconds of audio for chunk_index=%s bytes=%s",
                pcm_duration_seconds(pcm_bytes, self._sample_rate),
                chunk.index,
                len(pcm_bytes),
            )
            yield chunk.index, pcm_bytes

    def _synthesize_chunk(self, chunk_text_value: str, speaker_wav: str, settings: TTSSettings) -> bytes:
        assert self._tts is not None

        try:
            with self._synthesis_lock:
                with torch.inference_mode(), torch.no_grad():
                    waveform = self._tts.tts(
                        text=chunk_text_value,
                        speaker_wav=speaker_wav,
                        language=settings.language,
                        split_sentences=False,
                        temperature=settings.temperature,
                        top_p=settings.top_p,
                        speed=settings.speed,
                        length_penalty=settings.length_penalty,
                        repetition_penalty=settings.repetition_penalty,
                    )
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                if self.device == "cuda":
                    torch.cuda.empty_cache()
                raise RuntimeError("GPU out of memory during XTTS inference. Reduce text length or concurrency.") from exc
            raise

        if waveform is None:
            raise RuntimeError("Audio generation failed: XTTS returned None.")

        if hasattr(waveform, "tolist"):
            waveform = waveform.tolist()

        if len(waveform) == 0:
            raise RuntimeError("Audio generation failed: XTTS returned empty audio.")

        if self.device == "cuda":
            torch.cuda.empty_cache()

        pcm_bytes = pcm16le_bytes(waveform)
        if not pcm_bytes:
            raise RuntimeError("Audio generation failed: XTTS PCM buffer is empty.")
        return pcm_bytes

    def _validate_speaker_wav(self, speaker_wav: str) -> None:
        if not speaker_wav or not os.path.exists(speaker_wav):
            raise RuntimeError(f"Speaker WAV is missing: {speaker_wav}")
        metadata = validate_wav_file(speaker_wav, "Speaker WAV")
        LOGGER.info("Speaker WAV validated successfully: %s", metadata)

    def health(self) -> Dict[str, object]:
        gpu = None
        if torch.cuda.is_available():
            free_memory, total_memory = torch.cuda.mem_get_info()
            gpu = {
                "name": torch.cuda.get_device_name(0),
                "allocated_mb": round(torch.cuda.memory_allocated(0) / (1024 * 1024), 2),
                "reserved_mb": round(torch.cuda.memory_reserved(0) / (1024 * 1024), 2),
                "free_mb": round(free_memory / (1024 * 1024), 2),
                "total_mb": round(total_memory / (1024 * 1024), 2),
            }

        return {
            "status": "ok",
            "model_name": self.model_name,
            "model_loaded": self._tts is not None,
            "device": self.device,
            "sample_rate": self._sample_rate,
            "half_precision": self._using_half,
            "gpu": gpu,
            "defaults": asdict(TTSSettings()),
        }
