import os
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
TEMP_DIR = os.getenv("XTTS_TEMP_DIR", str(BASE_DIR / "tmp"))
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

DEFAULT_REFERENCE_SAMPLE_RATE = int(os.getenv("REFERENCE_SAMPLE_RATE", "16000"))
DEFAULT_OUTPUT_SAMPLE_RATE = int(os.getenv("OUTPUT_SAMPLE_RATE", "24000"))
MIN_WORDS_PER_CHUNK = int(os.getenv("MIN_WORDS_PER_CHUNK", "20"))
MAX_WORDS_PER_CHUNK = int(os.getenv("MAX_WORDS_PER_CHUNK", "40"))
MAX_TEXT_LENGTH = int(os.getenv("MAX_TEXT_LENGTH", "2500"))

XTTS_MODEL_NAME = os.getenv("XTTS_MODEL_NAME", "tts_models/multilingual/multi-dataset/xtts_v2")
XTTS_LANGUAGE = os.getenv("XTTS_LANGUAGE", "en")
XTTS_TEMPERATURE = float(os.getenv("XTTS_TEMPERATURE", "0.4"))
XTTS_TOP_P = float(os.getenv("XTTS_TOP_P", "0.85"))
XTTS_SPEED = float(os.getenv("XTTS_SPEED", "1.0"))
XTTS_LENGTH_PENALTY = float(os.getenv("XTTS_LENGTH_PENALTY", "1.0"))
XTTS_REPETITION_PENALTY = float(os.getenv("XTTS_REPETITION_PENALTY", "2.0"))

RVC_MODEL_PATH = os.getenv("RVC_MODEL_PATH", "")
RVC_INDEX_PATH = os.getenv("RVC_INDEX_PATH", "")
RVC_INFER_SCRIPT = os.getenv("RVC_INFER_SCRIPT", "")
RVC_PYTHON_BIN = os.getenv("RVC_PYTHON_BIN", "python")
RVC_PITCH = int(os.getenv("RVC_PITCH", "0"))
RVC_INDEX_RATE = float(os.getenv("RVC_INDEX_RATE", "0.75"))
RVC_PROTECT = float(os.getenv("RVC_PROTECT", "0.33"))
RVC_FILTER_RADIUS = int(os.getenv("RVC_FILTER_RADIUS", "3"))
RVC_HOP_LENGTH = int(os.getenv("RVC_HOP_LENGTH", "128"))
RVC_F0_METHOD = os.getenv("RVC_F0_METHOD", "rmvpe")

SUPPORTED_AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".flac", ".ogg"}
