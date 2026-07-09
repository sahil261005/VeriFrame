from pydantic import BaseModel, Field, EmailStr
from typing import Optional, Dict, Any, List
from datetime import datetime


# schemas for registration and login
class RegisterRequest(BaseModel):
    email: EmailStr
    password: str = Field(..., min_length=6)


class LoginRequest(BaseModel):
    email: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str


# schemas for analysis jobs
class JobStatusResponse(BaseModel):
    id: str
    status: str
    video_filename: str
    final_verdict: Optional[str] = None
    confidence: Optional[float] = None
    is_partial_analysis: bool = False
    created_at: datetime
    completed_at: Optional[datetime] = None

    class Config:
        from_attributes = True


class FullReportResponse(BaseModel):
    id: str
    status: str
    video_filename: str
    duration: float
    created_at: datetime
    completed_at: Optional[datetime] = None
    final_verdict: Optional[str] = None
    confidence: Optional[float] = None
    is_partial_analysis: bool = False
    report: Optional[Dict[str, Any]] = None
    thumbnails: Optional[List[Dict[str, Any]]] = None

    class Config:
        from_attributes = True
