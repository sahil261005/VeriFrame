from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, ForeignKey, Text
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
import uuid
from database import Base


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    password_hash = Column(String, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # relationship to jobs
    jobs = relationship("AnalysisJob", back_populates="user", cascade="all, delete-orphan")


class AnalysisJob(Base):
    __tablename__ = "analysis_jobs"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()), index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    status = Column(String, default="processing")  # processing, completed, failed
    video_filename = Column(String, nullable=False)
    duration = Column(Float, default=0.0)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    completed_at = Column(DateTime(timezone=True), nullable=True)
    
    # metrics
    final_verdict = Column(String, nullable=True)  # AUTHENTIC, MANIPULATED, UNCERTAIN
    confidence = Column(Float, nullable=True)
    is_partial_analysis = Column(Boolean, default=False)

    # report data stored as serialized JSON strings for SQLite compatibility
    report_json = Column(Text, nullable=True)
    flagged_frame_thumbnails = Column(Text, nullable=True)

    # relationship back to user
    user = relationship("User", back_populates="jobs")
