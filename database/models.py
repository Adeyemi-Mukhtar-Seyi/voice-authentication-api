from sqlalchemy import Column, Integer, String, LargeBinary, ForeignKey
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True)
    username = Column(String, unique=True, index=True)
    samples = relationship("VoiceSample", back_populates="user")

class VoiceSample(Base):
    __tablename__ = "voice_samples"
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    encrypted_mfcc = Column(LargeBinary)
    user = relationship("User", back_populates="samples")

