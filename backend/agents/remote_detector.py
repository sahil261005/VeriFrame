import os
import logging
import cv2

logger = logging.getLogger(__name__)

# shared client for an optional external detector service (POST /score and /faceswap with PNG frames,
# returns {"scores": [...]}). not included in this repo and not used in the deployed setup.
GEN_API_URL = os.environ.get("GEN_API_URL", "").rstrip("/")
GEN_API_TOKEN = os.environ.get("GEN_API_TOKEN", "")

# a sleeping free Space needs up to a minute or two to wake; warm_up() at server start usually avoids it
REMOTE_TIMEOUT = 100.0
# frames must be sent lossless: JPEG recompression (even quality 92) erases most of the detectors' signal
PNG_COMPRESSION = 3


def post_frames(endpoint, frames):
    """sends frames to the service and returns one probability per frame (same order)."""
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
    """wake the hosted Space (free Spaces sleep when idle) so the first real request is not slow."""
    if not GEN_API_URL:
        return
    try:
        import httpx
        httpx.get(f"{GEN_API_URL}/health", timeout=REMOTE_TIMEOUT)
        logger.info("detector service is awake")
    except Exception as e:
        logger.warning(f"could not reach detector service during warm-up: {e}")
