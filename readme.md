# Voice Cloning System

Production-oriented low-VRAM voice cloning stack using Flask + Coqui XTTS-v2 + Node.js streaming proxy.

## Structure

- `python/app.py`: Flask API with `/clone` and `/health`
- `python/tts_service.py`: singleton XTTS service and low-VRAM synthesis pipeline
- `python/audio_utils.py`: FFmpeg preprocessing, chunking, WAV streaming helpers
- `server/index.js`: Node.js multipart proxy that streams audio to the browser
- `server/test-client.js`: Node test script

## Python Setup

1. Create and activate Python 3.10+ virtual environment:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip setuptools wheel
```

2. Install PyTorch first.
   Pick the CUDA wheel that matches your driver/toolkit. PyTorch's install guide includes commands like:
   `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118`
   CPU fallback:
   `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu`

3. Install XTTS service dependencies:

```powershell
pip install -r python\requirements.txt
```

4. Install FFmpeg and add it to `PATH`.
   Verify:

```powershell
ffmpeg -version
```

5. Start the Flask API.
   Development:

```powershell
python python\app.py
```

   Production-friendly Windows service process:

```powershell
cd python
waitress-serve --listen=0.0.0.0:5001 app:app
```

## Node Setup

```powershell
cd server
npm install
npm start
```

Optional env vars:

- `PYTHON_API_URL=http://127.0.0.1:5001`
- `PORT=3000`

## API

### `POST /clone`

Multipart form fields:

- `text`: required input text
- `audio`: required reference voice file (`wav`, `mp3`, `m4a`, `flac`, `ogg`)
- `language`: optional, default `en`
- `temperature`: optional, default `0.65`
- `speed`: optional, default `1.0`
- `length_penalty`: optional, default `1.0`
- `repetition_penalty`: optional, default `2.0`
- `trim_silence`: optional, default `true`

Response:

- Chunked `audio/wav` stream

### `POST /health`

Returns JSON with:

- model load state
- device (`cuda` or `cpu`)
- sample rate
- half precision state
- GPU memory stats when CUDA is available

## cURL Example

```bash
curl -X POST http://127.0.0.1:5001/clone \
  -F "text=This is a production streaming XTTS test." \
  -F "audio=@voices/reference.wav" \
  -F "language=en" \
  -F "temperature=0.6" \
  --output cloned.wav
```

## Postman Example

Create a `POST` request to `http://127.0.0.1:3000/clone` with `Body -> form-data`:

- `text`: `Streaming voice cloning through the Node proxy.`
- `audio`: choose a local WAV or MP3 file
- `language`: `en`
- `temperature`: `0.65`
- `speed`: `1.0`

Set Postman to save the response as a file if you want to inspect the streamed WAV.

## Node Test Script

```powershell
cd server
npm run test:clone -- "This is a Node proxy integration test." "..\voices\reference.wav" ".\proxy-output.wav"
```

## Optimization Notes

- XTTS is loaded once with a singleton pattern.
- Inference uses `torch.inference_mode()` and `torch.no_grad()`.
- CUDA optimizations are enabled when available.
- GPU inference attempts half precision to reduce VRAM pressure.
- Reference audio is normalized by FFmpeg to `16kHz`, mono, high-pass filtered, low-pass filtered, and dynamically normalized.
- Text is chunked into roughly `20-40` words and synthesized sequentially to reduce OOM risk.
- WAV bytes are streamed chunk by chunk instead of writing full audio files.
- CUDA cache is released after each chunk to keep 4GB cards alive under load.

## Limitations

- A 4GB GPU is tight for XTTS-v2. Very long text, noisy reference audio, or concurrent requests can still trigger slowdowns or OOM.
- Chunking is required because XTTS conditioning and waveform generation grow memory use with longer input.
- Zero-shot cloning quality depends heavily on reference cleanliness, speaking style, room noise, and mic quality.
- Open-source XTTS can sound excellent, but perfect commercial-grade cloning consistency is not guaranteed.

## Recommended Reference Audio

- 6-20 seconds
- single speaker
- low room noise
- minimal music/reverb
- steady speaking pace
