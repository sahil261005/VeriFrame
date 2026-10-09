from pydantic import BaseModel, Field, EmailStr, field_validator
from typing import Optional, Dict, Any, List
from datetime import datetime


# schemas for registration and login
def _check_password_bytes(value: str) -> str:
    # bcrypt only reads the first 72 bytes and bcrypt 5 throws an error on longer ones so we reject it early
    if len(value.encode("utf-8")) > 72:
        raise ValueError("Password must be at most 72 bytes long")
    return value


class RegisterRequest(BaseModel):
    email: EmailStr = Field(..., max_length=254)
    password: str = Field(..., min_length=8, max_length=72)

    _password_bytes = field_validator("password")(_check_password_bytes)


class LoginRequest(BaseModel):
    email: str = Field(..., max_length=254)
    password: str = Field(..., max_length=128)


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
