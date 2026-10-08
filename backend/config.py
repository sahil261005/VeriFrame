import os

# simple configuration settings for our backend
# we read from environment variables or use safe defaults for local testing

DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///./veriframe.db")

# secret key for jwt tokens. in production, this should be a random string.
_PUBLISHED_DEFAULT_SECRET = "super-secret-key-change-this-in-production"  # was committed to the repo, so it is public
JWT_SECRET = os.environ.get("JWT_SECRET", "")
JWT_SECRET_IS_EPHEMERAL = not JWT_SECRET or JWT_SECRET == _PUBLISHED_DEFAULT_SECRET
if JWT_SECRET_IS_EPHEMERAL:
    # never sign tokens with a guessable key: use a random one (everyone is signed out whenever the server restarts)
    import secrets
    JWT_SECRET = secrets.token_urlsafe(48)
    print("WARNING: JWT_SECRET is not set (or is the published default). Using a random secret for this process; "
          "set JWT_SECRET to a long random value so logins survive restarts.", flush=True)
elif len(JWT_SECRET) < 32:
    print("WARNING: JWT_SECRET is shorter than 32 characters; use a longer random value.", flush=True)

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


def _allowed_cpus():
    # containers (Render, Docker) can see every core of the host but may only get a fraction of one;
    # the real limit is the cgroup CPU quota. falls back to the visible core count when there is no quota.
    try:
        with open("/sys/fs/cgroup/cpu.max") as f:  # cgroup v2: "<quota> <period>" or "max <period>"
            quota, period = f.read().split()[:2]
            if quota != "max":
                return int(quota) / int(period)
    except (OSError, ValueError):
        pass
    try:
        with open("/sys/fs/cgroup/cpu/cpu.cfs_quota_us") as fq, open("/sys/fs/cgroup/cpu/cpu.cfs_period_us") as fp:
            quota, period = int(fq.read()), int(fp.read())  # cgroup v1: quota -1 means unlimited
            if quota > 0:
                return quota / period
    except (OSError, ValueError):
        pass
    return os.cpu_count() or 1


# threads for ONNX Runtime / OpenCV / BLAS. without a cap each library starts one thread per *visible* core, and on a
# host limited to a fraction of a CPU those threads mostly fight each other (CV stage measured ~100x slower on Render).
CPU_THREADS = int(os.environ.get("ANALYSIS_THREADS") or max(1, min(os.cpu_count() or 1, int(_allowed_cpus()))))
