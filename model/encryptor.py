import os
from hashlib import sha256

from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes


def get_aes_key():
    raw_key = os.getenv("AES_KEY")
    if not raw_key:
        raise RuntimeError("AES_KEY environment variable is required")
    return sha256(raw_key.encode()).digest()


KEY = get_aes_key()


def encrypt(data: bytes) -> bytes:
    iv = os.urandom(16)
    cipher = Cipher(algorithms.AES(KEY), modes.CFB(iv), backend=default_backend())
    encryptor = cipher.encryptor()
    return iv + encryptor.update(data) + encryptor.finalize()


def decrypt(encrypted: bytes) -> bytes:
    if len(encrypted) < 16:
        raise ValueError("Invalid encrypted payload")
    iv = encrypted[:16]
    cipher = Cipher(algorithms.AES(KEY), modes.CFB(iv), backend=default_backend())
    decryptor = cipher.decryptor()
    return decryptor.update(encrypted[16:]) + decryptor.finalize()
