# Voice Cloning System

Two-stage streaming voice cloning pipeline:

1. XTTS-v2 generates natural base speech from text.
2. RVC converts that speech toward the target speaker identity.

## Files

- `python/app.py`
- `python/tts_service.py`
- `python/rvc_service.py`
- `python/audio_utils.py`
- `python/config.py`
- `server/index.js`
- `server/test-client.js`

## Python Setup

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip setuptools wheel
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r python\requirements.txt
```

CPU fallback:

```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

Install RVC runtime in the same environment or a separate one referenced by `RVC_PYTHON_BIN`. The service expects a callable RVC inference script that accepts:

```text
--input_path --output_path --model_path --index_path --pitch --index_rate --protect --filter_radius --hop_length --f0_method --speaker_wav
```

Typical setup:

1. Clone an RVC inference repo locally.
2. Install its dependencies in a Python environment.
3. Download an `.pth` voice model and optional `.index`.
4. Point the env vars below to those assets.

## FFmpeg

Install FFmpeg and add it to `PATH`.

```powershell
ffmpeg -version
```

## Environment Variables

```powershell
$env:XTTS_MODEL_NAME="tts_models/multilingual/multi-dataset/xtts_v2"
$env:XTTS_LANGUAGE="en"
$env:XTTS_TEMPERATURE="0.4"
$env:XTTS_TOP_P="0.85"
$env:XTTS_SPEED="1.0"
$env:RVC_INFER_SCRIPT="D:\rvc\infer_cli.py"
$env:RVC_MODEL_PATH="D:\rvc\models\target_voice.pth"
$env:RVC_INDEX_PATH="D:\rvc\models\target_voice.index"
$env:RVC_PYTHON_BIN="python"
$env:RVC_F0_METHOD="rmvpe"
```

## Run Python API

Development:

```powershell
python python\app.py
```

Production:

```powershell
cd python
waitress-serve --listen=0.0.0.0:5001 app:app
```

## Node Proxy

```powershell
cd server
npm install
npm start
```

Optional env vars:

- `PYTHON_API_URL=http://127.0.0.1:5001`
- `PORT=3000`

## API

### `POST /generate`

Multipart form-data:

- `text` required
- `audio` required
- `language` optional
- `temperature` optional, recommended `0.3-0.5`
- `top_p` optional, recommended `0.8-0.9`
- `speed` optional
- `length_penalty` optional
- `repetition_penalty` optional
- `trim_silence` optional
- `pitch` optional
- `index_rate` optional
- `protect` optional
- `filter_radius` optional
- `hop_length` optional
- `f0_method` optional

Response:

- streamed `audio/wav`

### `POST /health`

Returns XTTS load status, GPU memory stats, and RVC configuration state.

## cURL

```bash
curl -X POST http://127.0.0.1:5001/generate \
  -F "text=This is a high quality open source voice cloning test." \
  -F "audio=@voices/reference.wav" \
  -F "temperature=0.4" \
  -F "top_p=0.85" \
  --output cloned.wav
```

## Postman

Send `POST http://127.0.0.1:3000/generate` with `Body -> form-data`:

- `text`: your input text
- `audio`: a clean reference WAV
- `temperature`: `0.4`
- `top_p`: `0.85`
- `pitch`: `0`

Save the response to file.

## Node Test

```powershell
cd server
npm run test:clone -- "This is a Node proxy integration test." "..\voices\reference.wav" ".\proxy-output.wav"
```

## Notes

- 4GB VRAM is tight, so chunking and sequential inference are mandatory.
- Open-source XTTS + RVC can sound strong, but it will not perfectly match ElevenLabs consistency.
- RVC improves speaker similarity because it corrects the timbre of the XTTS output toward the target voice model.
- Quality depends heavily on clean reference audio with low noise, low echo, and stable speaking style.
