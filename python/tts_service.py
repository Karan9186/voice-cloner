import logging
import os
import threading
from dataclasses import asdict, dataclass
from typing import Dict, Generator, Optional

import torch
from TTS.api import TTS

from audio_utils import (
    DEFAULT_OUTPUT_SAMPLE_RATE,
    chunk_text,
    pcm16le_bytes,
    silence_pcm_bytes,
    text_word_count,
)


LOGGER = logging.getLogger(__name__)

MODEL_NAME = os.getenv("XTTS_MODEL_NAME", "tts_models/multilingual/multi-dataset/xtts_v2")
DEFAULT_LANGUAGE = os.getenv("XTTS_LANGUAGE", "en")
DEFAULT_TEMPERATURE = float(os.getenv("XTTS_TEMPERATURE", "0.65"))
DEFAULT_SPEED = float(os.getenv("XTTS_SPEED", "1.0"))
DEFAULT_LENGTH_PENALTY = float(os.getenv("XTTS_LENGTH_PENALTY", "1.0"))
DEFAULT_REPETITION_PENALTY = float(os.getenv("XTTS_REPETITION_PENALTY", "2.0"))


if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True


@dataclass
class SynthesisSettings:
    language: str = DEFAULT_LANGUAGE
    temperature: float = DEFAULT_TEMPERATURE
    speed: float = DEFAULT_SPEED
    length_penalty: float = DEFAULT_LENGTH_PENALTY
    repetition_penalty: float = DEFAULT_REPETITION_PENALTY


class TTSService:
    _instance: Optional["TTSService"] = None
    _instance_lock = threading.Lock()

    def __init__(self) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = MODEL_NAME
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
            output_sample_rate = getattr(synthesizer, "output_sample_rate", None)
            if isinstance(output_sample_rate, int) and output_sample_rate > 0:
                self._sample_rate = output_sample_rate

            self._tts = tts
            LOGGER.info(
                "XTTS model loaded. sample_rate=%s, half_precision=%s, device=%s",
                self._sample_rate,
                self._using_half,
                self.device,
            )

    def _enable_half_precision(self, tts: TTS) -> None:
        try:
            tts_model = getattr(getattr(tts, "synthesizer", None), "tts_model", None)
            if tts_model is not None:
                tts_model.half()
                self._using_half = True
        except Exception as exc:
            LOGGER.warning("Half precision could not be enabled cleanly: %s", exc)

    def build_settings(self, payload: Dict[str, str]) -> SynthesisSettings:
        settings = SynthesisSettings()
        if payload.get("language"):
            settings.language = payload["language"]
        if payload.get("temperature"):
            settings.temperature = float(payload["temperature"])
        if payload.get("speed"):
            settings.speed = float(payload["speed"])
        if payload.get("length_penalty"):
            settings.length_penalty = float(payload["length_penalty"])
        if payload.get("repetition_penalty"):
            settings.repetition_penalty = float(payload["repetition_penalty"])
        return settings

    def synthesize_stream(
        self,
        text: str,
        speaker_wav: str,
        settings: SynthesisSettings,
    ) -> Generator[bytes, None, None]:
        self.ensure_model_loaded()
        assert self._tts is not None

        chunks = chunk_text(text)
        if not chunks:
            raise ValueError("No valid text chunks were produced from the supplied text.")

        LOGGER.info(
            "Starting synthesis. words=%s chunks=%s device=%s language=%s",
            text_word_count(text),
            len(chunks),
            self.device,
            settings.language,
        )

        def generator() -> Generator[bytes, None, None]:
            first_chunk = True
            for chunk in chunks:
                LOGGER.info("Synthesizing chunk %s/%s (%s chars)", chunk.index + 1, len(chunks), len(chunk.text))
                pcm_bytes = self._synthesize_chunk(chunk.text, speaker_wav, settings)
                if not first_chunk:
                    yield silence_pcm_bytes(self._sample_rate)
                yield pcm_bytes
                first_chunk = False

        return generator()

    def _synthesize_chunk(self, chunk_text_value: str, speaker_wav: str, settings: SynthesisSettings) -> bytes:
        assert self._tts is not None
        with self._synthesis_lock:
            with torch.inference_mode(), torch.no_grad():
                waveform = self._tts.tts(
                    text=chunk_text_value,
                    speaker_wav=speaker_wav,
                    language=settings.language,
                    split_sentences=False,
                    temperature=settings.temperature,
                    speed=settings.speed,
                    length_penalty=settings.length_penalty,
                    repetition_penalty=settings.repetition_penalty,
                )

        if self.device == "cuda":
            torch.cuda.empty_cache()

        return pcm16le_bytes(waveform)

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
            "defaults": asdict(SynthesisSettings()),
        }
