import io
import math
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from typing import Generator, Iterable, List, Sequence

import numpy as np


DEFAULT_REFERENCE_SAMPLE_RATE = 16000
DEFAULT_OUTPUT_SAMPLE_RATE = 24000
MIN_WORDS_PER_CHUNK = 20
MAX_WORDS_PER_CHUNK = 40
SUPPORTED_AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".flac", ".ogg"}


class AudioProcessingError(RuntimeError):
    pass


@dataclass(frozen=True)
class AudioChunk:
    index: int
    text: str


def ensure_ffmpeg_installed() -> None:
    if shutil.which("ffmpeg"):
        return
    raise AudioProcessingError(
        "FFmpeg is not available on PATH. Install FFmpeg and ensure the ffmpeg binary is discoverable."
    )


def validate_audio_filename(filename: str) -> None:
    _, extension = os.path.splitext(filename or "")
    if extension.lower() not in SUPPORTED_AUDIO_EXTENSIONS:
        raise AudioProcessingError(
            f"Unsupported audio file type '{extension}'. Supported types: {sorted(SUPPORTED_AUDIO_EXTENSIONS)}"
        )


def save_upload_to_temp(upload_stream, suffix: str) -> str:
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        upload_stream.seek(0)
        shutil.copyfileobj(upload_stream, temp_file)
        return temp_file.name


def preprocess_reference_audio(
    input_path: str,
    output_dir: str | None = None,
    sample_rate: int = DEFAULT_REFERENCE_SAMPLE_RATE,
    trim_silence: bool = True,
) -> str:
    ensure_ffmpeg_installed()
    output_dir = output_dir or tempfile.gettempdir()
    os.makedirs(output_dir, exist_ok=True)

    cleaned_fd, cleaned_path = tempfile.mkstemp(prefix="clean_ref_", suffix=".wav", dir=output_dir)
    os.close(cleaned_fd)

    filters = [
        "highpass=f=80",
        "lowpass=f=8000",
        "dynaudnorm=f=150:g=15",
    ]
    if trim_silence:
        filters.append("silenceremove=start_periods=1:start_silence=0.15:start_threshold=-45dB")
        filters.append("areverse")
        filters.append("silenceremove=start_periods=1:start_silence=0.15:start_threshold=-45dB")
        filters.append("areverse")

    command = [
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
    ]

    try:
        subprocess.run(command, check=True, capture_output=True)
    except subprocess.CalledProcessError as exc:
        raise AudioProcessingError(exc.stderr.decode("utf-8", errors="ignore") or "FFmpeg preprocessing failed.") from exc

    return cleaned_path


def normalize_audio(audio: Sequence[float], peak: float = 0.95) -> np.ndarray:
    audio_array = np.asarray(audio, dtype=np.float32)
    if audio_array.size == 0:
        return audio_array
    max_value = float(np.max(np.abs(audio_array)))
    if max_value <= 1e-6:
        return audio_array
    scaled = audio_array * min(peak / max_value, 1.0)
    return np.clip(scaled, -1.0, 1.0)


def pcm16le_bytes(audio: Sequence[float]) -> bytes:
    normalized = normalize_audio(audio)
    pcm = (normalized * 32767.0).astype(np.int16)
    return pcm.tobytes()


def wav_stream_header(sample_rate: int, channels: int = 1, bits_per_sample: int = 16) -> bytes:
    byte_rate = sample_rate * channels * bits_per_sample // 8
    block_align = channels * bits_per_sample // 8
    max_uint32 = 0xFFFFFFFF
    header = io.BytesIO()
    header.write(b"RIFF")
    header.write(max_uint32.to_bytes(4, "little"))
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
    header.write(max_uint32.to_bytes(4, "little"))
    return header.getvalue()


def sentence_split(text: str) -> List[str]:
    normalized = re.sub(r"\s+", " ", text.strip())
    if not normalized:
        return []
    pieces = re.split(r"(?<=[.!?;:])\s+", normalized)
    return [piece.strip() for piece in pieces if piece.strip()]


def chunk_text(text: str, min_words: int = MIN_WORDS_PER_CHUNK, max_words: int = MAX_WORDS_PER_CHUNK) -> List[AudioChunk]:
    sentences = sentence_split(text)
    if not sentences:
        return []

    chunks: List[str] = []
    current_words: List[str] = []

    for sentence in sentences:
        words = sentence.split()
        if not words:
            continue

        if len(words) > max_words:
            if current_words:
                chunks.append(" ".join(current_words))
                current_words = []

            for index in range(0, len(words), max_words):
                chunks.append(" ".join(words[index : index + max_words]))
            continue

        projected_count = len(current_words) + len(words)
        if current_words and projected_count > max_words:
            chunks.append(" ".join(current_words))
            current_words = words[:]
        else:
            current_words.extend(words)

        if len(current_words) >= min_words:
            chunks.append(" ".join(current_words))
            current_words = []

    if current_words:
        chunks.append(" ".join(current_words))

    return [AudioChunk(index=index, text=chunk) for index, chunk in enumerate(chunks)]


def text_word_count(text: str) -> int:
    return len([token for token in re.split(r"\s+", text.strip()) if token])


def iter_with_tail_silence(chunks: Iterable[bytes], sample_rate: int, silence_ms: int = 40) -> Generator[bytes, None, None]:
    silence_samples = int(math.ceil(sample_rate * (silence_ms / 1000.0)))
    silence = (np.zeros(silence_samples, dtype=np.int16)).tobytes()
    first = True
    for chunk in chunks:
        if not first:
            yield silence
        yield chunk
        first = False


def silence_pcm_bytes(sample_rate: int, silence_ms: int = 40) -> bytes:
    silence_samples = int(math.ceil(sample_rate * (silence_ms / 1000.0)))
    return (np.zeros(silence_samples, dtype=np.int16)).tobytes()


def safe_unlink(*paths: str) -> None:
    for path in paths:
        if not path:
            continue
        try:
            if os.path.exists(path):
                os.unlink(path)
        except OSError:
            pass
