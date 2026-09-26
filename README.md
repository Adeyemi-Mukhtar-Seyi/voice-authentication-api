# Voice Authentication API

A FastAPI-based voice authentication system that registers users with voice samples, extracts MFCC-based audio features, encrypts the extracted features, trains a TensorFlow CNN classifier, and authenticates users from new voice recordings.

## Features

- Voice-based user registration with exactly three samples per user
- Audio preprocessing with FFmpeg/pydub
- Silence removal using WebRTC VAD
- Noise reduction with `noisereduce`
- MFCC, delta, and delta-delta feature extraction
- AES-256 encryption of stored MFCC features
- TensorFlow/Keras CNN voice classification
- Binary and multi-class classification support
- FastAPI REST endpoints and interactive Swagger documentation
- PostgreSQL support for production
- SQLite fallback for local development
- Docker deployment with FFmpeg included
- Render Blueprint configuration
- Health-check endpoint for deployment monitoring

## Architecture

```text
Audio Upload
    |
    v
FFmpeg / pydub
    |
    v
Noise Reduction + WebRTC VAD
    |
    v
MFCC + Delta + Delta-Delta
    |
    +--------------------+
    |                    |
    v                    v
AES-256 Encryption    TensorFlow CNN
    |                    |
    v                    v
PostgreSQL          Voice Prediction
```

## Project Structure

```text
voice_auth_update/
├── database/
│   └── models.py
├── model/
│   ├── audio_utils.py
│   ├── cnn.py
│   └── encryptor.py
├── static/
│   ├── home.html
│   └── index.html
├── voice_model/
│   ├── label_encoder.pkl
│   └── voice_cnn.h5
├── .dockerignore
├── .env.example
├── .gitignore
├── .python-version
├── Dockerfile
├── main.py
├── render.yaml
├── requirements.txt
└── run_server.py
```

## Requirements

- Python 3.11
- FFmpeg
- A supported TensorFlow environment
- PostgreSQL for production, or SQLite for local development

## Local Setup

### 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd voice_auth_update
```

### 2. Create a virtual environment

Windows PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Install FFmpeg

FFmpeg must be installed and available on your system `PATH` because uploaded MP3/WebM/OGG files are converted to WAV before feature extraction.

### 5. Configure environment variables

Copy `.env.example` to `.env` and provide a strong AES key and training API key.

The application supports:

```text
DATABASE_URL
AES_KEY
TRAINING_API_KEY
FRONTEND_ORIGINS
ENABLE_DEBUG_AUDIO
LABEL_ENCODER_PATH
MODEL_PATH
```

For local development, the default database is SQLite:

```text
DATABASE_URL=sqlite:///./voice_auth.db
```

### 6. Start the API

```bash
python run_server.py
```

Or:

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Open:

- Application: `http://localhost:8000`
- Swagger UI: `http://localhost:8000/docs`
- Health check: `http://localhost:8000/health`

## API Endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/` | Serves the application frontend |
| GET | `/home/` | Serves the authenticated home page |
| GET | `/health` | Service health check |
| POST | `/register/` | Registers a user with three voice samples |
| POST | `/train/` | Trains the CNN model; requires `X-API-Key` |
| POST | `/login/` | Authenticates a user using a voice sample |
| GET | `/users/` | Lists registered users and sample counts |
| POST | `/debug-audio/` | Disabled by default; development diagnostics only |

## Model Training

Training requires at least two distinct users and at least two valid voice samples per user.

The training endpoint requires the `X-API-Key` HTTP header:

```http
X-API-Key: YOUR_TRAINING_API_KEY
```

The resulting label encoder and CNN model are written to the configured model directory.

## Security Notes

- AES encryption requires `AES_KEY` to be supplied through the environment.
- Secrets must not be committed to GitHub.
- Raw voice datasets and local databases are excluded from Git.
- Model training is protected by `TRAINING_API_KEY`.
- The debug audio endpoint is disabled unless explicitly enabled.
- Production CORS origins should be restricted to trusted frontend origins.
- Voice recordings and extracted biometric features should be handled according to applicable privacy and data-protection requirements.

## Docker

Build the image:

```bash
docker build -t voice-authentication-api .
```

Run it locally:

```bash
docker run --rm -p 8000:8000 \
  -e AES_KEY="replace-with-a-long-random-secret" \
  -e TRAINING_API_KEY="replace-with-a-long-random-training-key" \
  -e DATABASE_URL="sqlite:////tmp/voice_auth.db" \
  voice-authentication-api
```

The Docker image installs FFmpeg and `libsndfile1`, so the audio-processing system dependencies are included in the deployment image.

## Render Deployment

The repository includes `render.yaml` for Render Blueprint deployment.

The production architecture is:

```text
Render Web Service
        |
        +-- FastAPI + TensorFlow + FFmpeg
        |
        +-- Render Postgres
```

The Render service receives its database connection string through `DATABASE_URL`. Secrets are generated by Render rather than committed to source control.

After deployment, verify:

```text
https://YOUR-SERVICE.onrender.com/health
https://YOUR-SERVICE.onrender.com/docs
```

## Data and Model Files

The repository contains the trained inference model and label encoder required by the application. It intentionally does not contain the local SQLite database, raw training recordings, Python virtual environment, notebook checkpoints, or temporary training checkpoints.

## Limitations

This project is a machine-learning voice classification system. Classification confidence is model output, not a cryptographic proof of speaker identity. Production deployments should evaluate the model against representative data, false-acceptance/false-rejection rates, replay/spoofing scenarios, and the privacy requirements applicable to the deployment environment.

## License

Add the project's license here before publishing the repository publicly.
