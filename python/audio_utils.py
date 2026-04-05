import io
import logging
import math
import os
import re
import shutil
import subprocess
import tempfile
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Sequence

import numpy as np

from config import (
    DEFAULT_OUTPUT_SAMPLE_RATE,
    DEFAULT_REFERENCE_SAMPLE_RATE,
    MAX_WORDS_PER_CHUNK,
    MIN_WORDS_PER_CHUNK,
    SUPPORTED_AUDIO_EXTENSIONS,
)


class AudioProcessingError(RuntimeError):
    pass


@dataclass(frozen=True)
class AudioChunk:
    index: int
    text: str


@dataclass(frozen=True)
class VoiceChunkArtifacts:
    pcm_bytes: bytes
    input_wav_path: str
    output_wav_path: str
    resampled_output_path: str

    def cleanup_paths(self) -> tuple[str, str, str]:
        return self.input_wav_path, self.output_wav_path, self.resampled_output_path


def ensure_ffmpeg_installed() -> None:
    if shutil.which("ffmpeg"):
        return
    raise AudioProcessingError("FFmpeg is not available on PATH.")


def validate_audio_filename(filename: str) -> None:
    _, extension = os.path.splitext(filename or "")
    if extension.lower() not in SUPPORTED_AUDIO_EXTENSIONS:
        raise AudioProcessingError(
            f"Unsupported audio file type '{extension}'. Supported types: {sorted(SUPPORTED_AUDIO_EXTENSIONS)}"
        )


def save_upload_to_temp(upload_stream, filename: str, output_dir: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    suffix = Path(filename or "reference.wav").suffix or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix, dir=output_dir) as temp_file:
        upload_stream.seek(0)
        shutil.copyfileobj(upload_stream, temp_file)
        temp_file.flush()
        return temp_file.name


def preprocess_reference_audio(
    input_path: str,
    output_dir: str,
    sample_rate: int = DEFAULT_REFERENCE_SAMPLE_RATE,
    trim_silence: bool = True,
) -> str:
    ensure_ffmpeg_installed()
    os.makedirs(output_dir, exist_ok=True)
    logger = logging.getLogger(__name__)
    logger.info(
        "Audio preprocessing started. input_path=%s trim_silence=%s target_sample_rate=%s",
        input_path,
        trim_silence,
        sample_rate,
    )

    cleaned_fd, cleaned_path = tempfile.mkstemp(prefix="clean_ref_", suffix=".wav", dir=output_dir)
    os.close(cleaned_fd)

    filters = [
        "highpass=f=80",
        "lowpass=f=8000",
        "afftdn",
        "loudnorm=I=-16:TP=-1.5:LRA=11",
        "dynaudnorm=f=120:g=10",
    ]
    if trim_silence:
        filters.extend(
            [
                "silenceremove=start_periods=1:start_silence=0.12:start_threshold=-45dB",
                "areverse",
                "silenceremove=start_periods=1:start_silence=0.12:start_threshold=-45dB",
                "areverse",
            ]
        )

    run_subprocess(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            input_path,
            "-af",
            ",".join(filters),
            "-ar",
            str(sample_rate),
            "-ac",
            "1",
            cleaned_path,
        ],
        "Reference audio preprocessing failed.",
    )
    metadata = inspect_wav_file(cleaned_path)
    logger.info("Audio loaded successfully after preprocessing: %s", metadata)
    validate_wav_file(cleaned_path, "Cleaned reference audio")
    return cleaned_path


def normalize_audio(audio: Sequence[float], peak: float = 0.95) -> np.ndarray:
    audio_array = np.asarray(audio, dtype=np.float32)
    if audio_array.size == 0:
        return audio_array
    max_value = float(np.max(np.abs(audio_array)))
    if max_value <= 1e-6:
        return audio_array
    return np.clip(audio_array * min(peak / max_value, 1.0), -1.0, 1.0)


def pcm16le_bytes(audio: Sequence[float]) -> bytes:
    normalized = normalize_audio(audio)
    return (normalized * 32767.0).astype(np.int16).tobytes()


def pcm_duration_seconds(pcm_bytes: bytes, sample_rate: int, sample_width: int = 2, channels: int = 1) -> float:
    if sample_rate <= 0 or sample_width <= 0 or channels <= 0:
        return 0.0
    frame_width = sample_width * channels
    if frame_width <= 0:
        return 0.0
    return len(pcm_bytes) / float(sample_rate * frame_width)


def wav_stream_header(sample_rate: int, channels: int = 1, bits_per_sample: int = 16) -> bytes:
    byte_rate = sample_rate * channels * bits_per_sample // 8
    block_align = channels * bits_per_sample // 8
    placeholder_size = 0xFFFFFFFF
    header = io.BytesIO()
    header.write(b"RIFF")
    header.write(placeholder_size.to_bytes(4, "little"))
    header.write(b"WAVE")
    header.write(b"fmt ")
    header.write((16).to_bytes(4, "little"))
    header.write((1).to_bytes(2, "little"))
    header.write(channels.to_bytes(2, "little"))
    header.write(sample_rate.to_bytes(4, "little"))
    header.write(byte_rate.to_bytes(4, "little"))
    header.write(block_align.to_bytes(2, "little"))
    header.write(bits_per_sample.to_bytes(2, "little"))
    header.write(b"data")
    header.write(placeholder_size.to_bytes(4, "little"))
    return header.getvalue()


def sentence_split(text: str) -> List[str]:
    normalized = re.sub(r"\s+", " ", text.strip())
    if not normalized:
        return []
    return [piece.strip() for piece in re.split(r"(?<=[.!?;:])\s+", normalized) if piece.strip()]


def chunk_text(text: str, min_words: int = MIN_WORDS_PER_CHUNK, max_words: int = MAX_WORDS_PER_CHUNK) -> List[AudioChunk]:
    sentences = sentence_split(text)
    chunks: List[str] = []
    buffer: List[str] = []

    for sentence in sentences:
        words = sentence.split()
        if not words:
            continue

        if len(words) > max_words:
            if buffer:
                chunks.append(" ".join(buffer))
                buffer = []
            for start in range(0, len(words), max_words):
                chunks.append(" ".join(words[start : start + max_words]))
            continue

        if buffer and len(buffer) + len(words) > max_words:
            chunks.append(" ".join(buffer))
            buffer = words[:]
        else:
            buffer.extend(words)

        if len(buffer) >= min_words:
            chunks.append(" ".join(buffer))
            buffer = []

    if buffer:
        chunks.append(" ".join(buffer))

    return [AudioChunk(index=index, text=value) for index, value in enumerate(chunks)]


def text_word_count(text: str) -> int:
    return len([token for token in re.split(r"\s+", text.strip()) if token])


def silence_pcm_bytes(sample_rate: int, silence_ms: int = 35) -> bytes:
    silence_samples = int(math.ceil(sample_rate * (silence_ms / 1000.0)))
    return (np.zeros(silence_samples, dtype=np.int16)).tobytes()


def write_pcm_wav(path: str, pcm_bytes: bytes, sample_rate: int) -> None:
    if not pcm_bytes:
        raise AudioProcessingError("Refusing to write empty PCM audio to WAV.")
    with wave.open(path, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm_bytes)
    validate_wav_file(path, "Written WAV audio")


def read_wav_pcm(path: str) -> tuple[int, bytes]:
    with wave.open(path, "rb") as wav_file:
        sample_rate = wav_file.getframerate()
        pcm_bytes = wav_file.readframes(wav_file.getnframes())
    return sample_rate, pcm_bytes


def inspect_wav_file(path: str) -> dict[str, Any]:
    if not os.path.exists(path):
        raise AudioProcessingError(f"WAV file does not exist: {path}")

    with wave.open(path, "rb") as wav_file:
        channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        sample_rate = wav_file.getframerate()
        frame_count = wav_file.getnframes()

    duration_seconds = frame_count / float(sample_rate) if sample_rate > 0 else 0.0
    return {
        "path": path,
        "size_bytes": os.path.getsize(path),
        "channels": channels,
        "sample_width": sample_width,
        "sample_rate": sample_rate,
        "frame_count": frame_count,
        "duration_seconds": round(duration_seconds, 4),
    }


def validate_wav_file(path: str, label: str) -> dict[str, Any]:
    metadata = inspect_wav_file(path)
    if metadata["size_bytes"] <= 44:
        raise AudioProcessingError(f"{label} is empty or contains only a WAV header: {path}")
    if metadata["frame_count"] <= 0:
        raise AudioProcessingError(f"{label} contains zero frames: {path}")
    if metadata["duration_seconds"] <= 0:
        raise AudioProcessingError(f"{label} duration is zero seconds: {path}")
    return metadata


def run_subprocess(command: list[str], error_message: str) -> None:
    logging.getLogger(__name__).info("Running subprocess: %s", " ".join(command))
    try:
        subprocess.run(command, check=True, capture_output=True)
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.decode("utf-8", errors="ignore")
        raise AudioProcessingError(f"{error_message} {stderr}".strip()) from exc


def ffmpeg_resample(input_path: str, output_path: str, sample_rate: int = DEFAULT_OUTPUT_SAMPLE_RATE) -> None:
    run_subprocess(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            input_path,
            "-ar",
            str(sample_rate),
            "-ac",
            "1",
            output_path,
        ],
        "Audio resampling failed.",
    )


def safe_unlink(*paths: str) -> None:
    for path in paths:
        if not path:
            continue
        try:
            if os.path.exists(path):
                os.unlink(path)
        except OSError:
            pass
