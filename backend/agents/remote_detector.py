import os
import logging
import cv2

logger = logging.getLogger(__name__)

# client for an optional external detector service. you POST png frames to /score or /faceswap
# and get back {"scores": [...]}. the service isnt in this repo and the deployed version doesnt use it
GEN_API_URL = os.environ.get("GEN_API_URL", "").rstrip("/")
GEN_API_TOKEN = os.environ.get("GEN_API_TOKEN", "")

# free spaces go to sleep and can take a minute or two to wake up, calling warm_up() on startup mostly fixes this
REMOTE_TIMEOUT = 100.0
# send frames as png not jpeg, jpeg compression even at quality 92 wipes out most of what the detectors look for
PNG_COMPRESSION = 3


def post_frames(endpoint, frames):
    """sends frames to the service and gets back one probability per frame in the same order"""
    import httpx
    files = []
    for i, f in enumerate(frames):
        ok, buf = cv2.imencode(".png", f["image"], [cv2.IMWRITE_PNG_COMPRESSION, PNG_COMPRESSION])
        if not ok:
            raise RuntimeError("could not encode frame")
        files.append(("files", (f"frame{i}.png", buf.tobytes(), "image/png")))
    headers = {"Authorization": f"Bearer {GEN_API_TOKEN}"} if GEN_API_TOKEN else {}

    last_error = None
    for attempt in range(2):
        try:
            r = httpx.post(f"{GEN_API_URL}{endpoint}", files=files, headers=headers, timeout=REMOTE_TIMEOUT)
            r.raise_for_status()
            scores = r.json()["scores"]
            if len(scores) != len(frames):
                raise RuntimeError("detector service returned a different number of scores")
            return [float(x) for x in scores]
        except Exception as e:
            last_error = e
            logger.warning(f"detector request {endpoint} failed (attempt {attempt + 1}): {e}")
    raise RuntimeError(f"detector service unavailable: {last_error}")


def warm_up():
    """ping the hosted space so its awake before the first real request"""
    if not GEN_API_URL:
        return
    try:
        import httpx
        httpx.get(f"{GEN_API_URL}/health", timeout=REMOTE_TIMEOUT)
        logger.info("detector service is awake")
    except Exception as e:
        logger.warning(f"could not reach detector service during warm-up: {e}")
