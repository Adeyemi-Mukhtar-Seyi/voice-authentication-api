import io
import os
import tempfile

import numpy as np
import librosa
import webrtcvad
import noisereduce as nr
from pydub import AudioSegment


def convert_mp3_to_wav(file_bytes):
    """Convert various audio formats to WAV bytes."""
    formats = ["mp3", "wav", "webm", "ogg", None]  # Try None last to auto-detect

    for fmt in formats:
        try:
            audio = AudioSegment.from_file(io.BytesIO(file_bytes), format=fmt)
            break
        except:
            continue
    else:
        raise ValueError("Unable to process audio format")

    if audio.channels > 1:
        audio = audio.set_channels(1)

    audio = audio.set_frame_rate(16000)

    wav_io = io.BytesIO()
    audio.export(wav_io, format="wav")
    return wav_io.getvalue()


def remove_silence(y, sr, frame_duration_ms=30):
    """Remove silent segments using WebRTC VAD."""
    vad = webrtcvad.Vad(3)
    frame_length = int(sr * frame_duration_ms / 1000)
    voiced_frames = []

    for i in range(0, len(y) - frame_length, frame_length):
        frame = y[i:i + frame_length]
        if len(frame) < frame_length:
            continue
        pcm_bytes = (frame * 32768).astype(np.int16).tobytes()
        if vad.is_speech(pcm_bytes, sample_rate=sr):
            voiced_frames.extend(frame)

    return np.array(voiced_frames)


def extract_mfcc(wav_bytes, sr=16000, n_mfcc=13, max_len=450, padding_mode='constant'):
    """Extract MFCC features from WAV bytes."""
    try:
        try:
            y, sr_actual = librosa.load(io.BytesIO(wav_bytes), sr=sr)
        except:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                tmp_file.write(wav_bytes)
                tmp_file.flush()
                try:
                    y, sr_actual = librosa.load(tmp_file.name, sr=sr)
                finally:
                    os.unlink(tmp_file.name)

        y = nr.reduce_noise(y=y, sr=sr_actual)
        y = remove_silence(y, sr_actual)

        if len(y) == 0:
            raise ValueError("Audio became empty after silence removal")

        if np.max(np.abs(y)) < 1e-6:
            y += np.random.normal(0, 1e-6, y.shape)

        mfcc = librosa.feature.mfcc(y=y, sr=sr_actual, n_mfcc=n_mfcc)
        delta = librosa.feature.delta(mfcc)
        delta2 = librosa.feature.delta(mfcc, order=2)

        current_len = mfcc.shape[1]

        if current_len < max_len:
            pad_width = max_len - current_len
            pad_config = ((0, 0), (0, pad_width))
            mfcc = np.pad(mfcc, pad_width=pad_config, mode=padding_mode)
            delta = np.pad(delta, pad_width=pad_config, mode=padding_mode)
            delta2 = np.pad(delta2, pad_width=pad_config, mode=padding_mode)
        else:
            start_idx = (current_len - max_len) // 2
            end_idx = start_idx + max_len
            mfcc = mfcc[:, start_idx:end_idx]
            delta = delta[:, start_idx:end_idx]
            delta2 = delta2[:, start_idx:end_idx]

        features = np.stack([mfcc, delta, delta2], axis=-1)
        features = np.transpose(features, (1, 0, 2))
        features = (features - np.mean(features)) / (np.std(features) + 1e-8)

        return features.astype(np.float32)

    except Exception as e:
        print(f"Error extracting MFCC: {e}")
        return np.zeros((max_len, n_mfcc, 3), dtype=np.float32)


def process_audio_file(file_bytes):
    try:
        if not file_bytes:
            raise ValueError("Empty audio file")

        wav_bytes = convert_mp3_to_wav(file_bytes)
        print(f"[DEBUG] WAV size: {len(wav_bytes)}")

        mfcc = extract_mfcc(wav_bytes)
        print(f"[DEBUG] MFCC shape: {mfcc.shape}, dtype: {mfcc.dtype}, var: {np.var(mfcc)}")

        if mfcc.shape != (450, 13, 3) or np.all(mfcc == 0) or np.var(mfcc) < 1e-10:
            raise ValueError("Invalid MFCC")

        return mfcc

    except Exception as e:
        print(f"Error processing audio: {e}")
        return np.zeros((450, 13, 3), dtype=np.float32)


def validate_audio_features(features):
    if features is None:
        return False
    if features.shape != (450, 13, 3):
        return False
    if np.all(features == 0):
        return False
    if np.any(np.isnan(features)) or np.any(np.isinf(features)):
        return False
    if np.var(features) < 1e-10:
        return False
    return True

