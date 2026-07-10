import os

# simple configuration settings for our backend
# we read from environment variables or use safe defaults for local testing

DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///./veriframe.db")

# secret key for jwt tokens. in production, this should be a random string.
JWT_SECRET = os.environ.get("JWT_SECRET", "super-secret-key-change-this-in-production")

# default token expiration (30 minutes)
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# folder to temporarily save uploaded videos before processing
UPLOAD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "temp_uploads")

# make sure upload directory exists
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)

# CORS origins for security control (handles both plural 'CORS_ORIGINS' and singular 'CORS_ORIGIN' env variables)
raw_origins = os.environ.get("CORS_ORIGINS") or os.environ.get("CORS_ORIGIN") or "http://localhost:5173"
CORS_ORIGINS = [origin.strip().rstrip("/") for origin in raw_origins.split(",")]
