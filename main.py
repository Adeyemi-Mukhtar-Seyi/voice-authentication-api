from pathlib import Path
import logging
import os

import joblib
import numpy as np
import tensorflow as tf
from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

from database.models import Base, User, VoiceSample
from model.audio_utils import process_audio_file
from model.cnn import build_binary_model, build_model
from model.encryptor import decrypt, encrypt

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
MODEL_DIR = BASE_DIR / "voice_model"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Voice Authentication API", version="1.0.0")

frontend_origins = [
    origin.strip()
    for origin in os.getenv("FRONTEND_ORIGINS", "http://localhost:8000").split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=frontend_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"]
)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


def get_database_url() -> str:
    database_url = os.getenv("DATABASE_URL")
    if database_url:
        # Some providers still expose the legacy postgres:// scheme.
        if database_url.startswith("postgres://"):
            database_url = "postgresql+psycopg2://" + database_url[len("postgres://"):]
        elif database_url.startswith("postgresql://"):
            database_url = "postgresql+psycopg2://" + database_url[len("postgresql://"):]
        return database_url

    return f"sqlite:///{BASE_DIR / 'voice_auth.db'}"


DATABASE_URL = get_database_url()
connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
engine = create_engine(DATABASE_URL, connect_args=connect_args, pool_pre_ping=True)
Base.metadata.create_all(bind=engine)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

LABEL_ENCODER_PATH = Path(os.getenv("LABEL_ENCODER_PATH", str(MODEL_DIR / "label_encoder.pkl")))
MODEL_PATH = Path(os.getenv("MODEL_PATH", str(MODEL_DIR / "voice_cnn.h5")))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def require_training_api_key(x_api_key: str | None = Header(default=None)):
    expected_key = os.getenv("TRAINING_API_KEY")
    if not expected_key:
        raise HTTPException(status_code=503, detail="Model training is not configured")
    if not x_api_key or x_api_key != expected_key:
        raise HTTPException(status_code=401, detail="Invalid training API key")


@app.get("/")
async def serve_frontend():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/home/")
async def home_page():
    return FileResponse(STATIC_DIR / "home.html")


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/register/")
async def register(
    username: str = Form(...),
    files: list[UploadFile] = File(...),
    db: Session = Depends(get_db),
):
    username = username.strip()
    if not username:
        raise HTTPException(status_code=400, detail="Username is required")

    existing_user = db.query(User).filter(User.username == username).first()
    if existing_user:
        raise HTTPException(status_code=400, detail="Username already exists")

    if len(files) != 3:
        raise HTTPException(status_code=400, detail="Exactly 3 voice samples required")

    user = User(username=username)
    db.add(user)
    db.commit()
    db.refresh(user)

    try:
        for i, file in enumerate(files):
            content = await file.read()
            mfcc = process_audio_file(content)

            if np.all(mfcc == 0):
                raise HTTPException(status_code=400, detail=f"Failed to process audio sample {i + 1}")

            encrypted = encrypt(mfcc.tobytes())
            db.add(VoiceSample(user_id=user.id, encrypted_mfcc=encrypted))

        db.commit()
        return {"message": "User registered successfully", "success": True}
    except HTTPException:
        db.rollback()
        raise
    except Exception as exc:
        db.rollback()
        logger.exception("Registration failed")
        raise HTTPException(status_code=500, detail="Registration failed") from exc


@app.post("/train/")
def train_model(
    db: Session = Depends(get_db),
    _: None = Depends(require_training_api_key),
):
    users = db.query(User).all()
    users_with_samples = [user for user in users if user.samples]

    if len(users_with_samples) < 2:
        raise HTTPException(status_code=400, detail="At least 2 users with voice samples are required")

    X, y = [], []
    for user in users_with_samples:
        for sample in user.samples:
            try:
                decrypted = decrypt(sample.encrypted_mfcc)
                mfcc = np.frombuffer(decrypted, dtype=np.float32).reshape(450, 13, 3)
                X.append(mfcc)
                y.append(user.username)
            except Exception as exc:
                logger.warning("Failed to process sample for %s: %s", user.username, exc)

    if not X:
        raise HTTPException(status_code=400, detail="No valid samples found")

    X = np.asarray(X)
    y = np.asarray(y)
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    if len(le.classes_) < 2:
        raise HTTPException(status_code=400, detail="At least 2 distinct users are required for training")

    class_counts = np.bincount(y_enc)
    if np.min(class_counts) < 2:
        raise HTTPException(status_code=400, detail="Each user needs at least 2 valid voice samples for training")

    try:
        X_train, X_val, y_train, y_val = train_test_split(
            X,
            y_enc,
            test_size=0.3,
            random_state=42,
            stratify=y_enc,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"Training data is insufficient for validation split: {exc}") from exc

    LABEL_ENCODER_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(le, LABEL_ENCODER_PATH)

    logger.info("Train label distribution: %s", np.bincount(y_train))
    logger.info("Val label distribution: %s", np.bincount(y_val))

    if len(le.classes_) == 2:
        model = build_binary_model((450, 13, 3))
    else:
        model = build_model((450, 13, 3), len(le.classes_))

    checkpoint_path = MODEL_DIR / "best_model.h5"
    lr_reduce = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3)
    checkpoint = ModelCheckpoint(checkpoint_path, save_best_only=True)

    model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=90,
        batch_size=32,
        verbose=1,
        callbacks=[lr_reduce, checkpoint],
    )
    model.save(MODEL_PATH)

    return {"message": "Model trained successfully", "success": True}


@app.post("/login/")
async def login(username: str = Form(...), file: UploadFile = File(...)):
    if not MODEL_PATH.exists() or not LABEL_ENCODER_PATH.exists():
        raise HTTPException(status_code=400, detail="Model not trained yet. Please train the model first.")

    try:
        content = await file.read()
        mfcc = process_audio_file(content)
        if np.all(mfcc == 0):
            raise HTTPException(status_code=400, detail="Failed to process audio file")

        model = tf.keras.models.load_model(MODEL_PATH)
        le = joblib.load(LABEL_ENCODER_PATH)
        preds = model.predict(np.expand_dims(mfcc, axis=0), verbose=0)

        if len(le.classes_) == 2 and preds.shape[-1] == 1:
            probability_class_1 = float(preds[0][0])
            pred_class = 1 if probability_class_1 >= 0.5 else 0
            confidence = probability_class_1 if pred_class == 1 else 1.0 - probability_class_1
        else:
            pred_class = int(np.argmax(preds[0]))
            confidence = float(np.max(preds[0]))

        pred_label = le.inverse_transform([pred_class])[0]

        if pred_label == username and confidence >= 0.75:
            return {
                "message": "Login successful",
                "success": True,
                "confidence": confidence,
                "predicted_user": pred_label,
                "redirect": f"/home/?user={username}",
            }

        return {
            "message": "Authentication failed",
            "success": False,
            "confidence": confidence,
            "predicted_user": pred_label,
        }

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Login failed")
        raise HTTPException(status_code=500, detail="Login failed") from exc


@app.get("/users/")
def get_users(db: Session = Depends(get_db)):
    users = db.query(User).all()
    return {
        "users": [
            {"id": user.id, "username": user.username, "samples_count": len(user.samples)}
            for user in users
        ]
    }


@app.post("/debug-audio/")
async def debug_audio(file: UploadFile = File(...)):
    if os.getenv("ENABLE_DEBUG_AUDIO", "false").lower() != "true":
        raise HTTPException(status_code=404, detail="Not found")

    content = await file.read()
    return {
        "filename": file.filename,
        "content_type": file.content_type,
        "size": len(content),
        "first_bytes": content[:20].hex() if len(content) > 20 else content.hex(),
    }
