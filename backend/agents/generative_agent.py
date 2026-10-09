import os
import logging
import threading
import numpy as np
import cv2
from concurrent.futures import ThreadPoolExecutor
import agents.remote_detector as remote_detector

logger = logging.getLogger(__name__)

# detects videos that are fully AI generated, like sora, kling, runway or veo output.
# this is different from the face swap ViT in visual_agent, here we score the whole frame.
# model is a SwinV2 classifier (haywoodsloan/ai-image-detector-deploy), 195M params
#
# its too big for the 512MB render instance so it can run 2 ways:
#   1. remote: an external service, POST png frames to /score and get {"scores": [...]} back.
#      set GEN_API_URL and GEN_API_TOKEN if it needs a token. the service isnt in this repo
#   2. local: on a dev machine with enough RAM, uses the INT8 onnx from export_generative_model.py at GEN_MODEL_PATH
# if neither is set up the agent just says "skipped" and the other agents decide the verdict
DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "model_gen_onnx", "model_quantized.onnx")
MODEL_PATH = os.environ.get("GEN_MODEL_PATH", DEFAULT_MODEL_PATH)
GEN_API_URL = remote_detector.GEN_API_URL

INPUT_SIZE = 256
# same preprocessing the original model used, bicubic resize and imagenet mean/std
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
ARTIFICIAL_INDEX = 0  # label 0 is artificial, 1 is real

_session = None
_loaded = False
_lock = threading.Lock()


class _Remote:
    """just a marker so we know to use the remote detector"""


def load_model():
    """returns the local onnx session or the remote marker, None if nothing is set up"""
    global _session, _loaded
    if _loaded:
        return _session
    with _lock:
        if _loaded:
            return _session
        try:
            if os.path.exists(MODEL_PATH):
                import onnxruntime as ort
                _session = ort.InferenceSession(MODEL_PATH)
                logger.info(f"generative-video detector loaded locally ({os.path.getsize(MODEL_PATH) / (1024*1024):.0f} MB)")
            elif GEN_API_URL:
                _session = _Remote()
                logger.info(f"generative-video detector will use remote service {GEN_API_URL}")
            else:
                logger.warning("generative detector not configured (no GEN_API_URL or local model); agent will be skipped")
        except Exception as e:
            logger.warning(f"failed to load generative detector: {e}")
            _session = None
        _loaded = True
        return _session


def score_frames_remote(frames):
    return remote_detector.post_frames("/score", frames)


def _preprocess(img_bgr):
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_CUBIC)
    x = (rgb.astype(np.float32) / 255.0 - _MEAN) / _STD
    return np.expand_dims(np.transpose(x, (2, 0, 1)), axis=0)


def score_frame(img_bgr, session):
    """chance from 0 to 1 that one frame is AI generated, None if it fails"""
    try:
        logits = session.run(None, {session.get_inputs()[0].name: _preprocess(img_bgr)})[0][0]
        e = np.exp(logits - np.max(logits))
        return float((e / e.sum())[ARTIFICIAL_INDEX])
    except Exception as e:
        logger.warning(f"generative inference failed: {e}")
        return None


def aggregate(frame_scores):
    """
    turns per frame scores into one video score. the detector misses some frames in fake clips because of
    blur or compression, so we average the top half instead of all frames. on real videos this still kept false alarms low
    """
    if not frame_scores:
        return 0.0
    ordered = sorted(frame_scores, reverse=True)
    top = ordered[:max(1, (len(ordered) + 1) // 2)]
    return float(np.mean(top))


def analyze_frames(frames, session):
    """returns (video_score, per_frame_results), raises RuntimeError if every frame failed"""
    if isinstance(session, _Remote):
        scores = score_frames_remote(frames)
    else:
        with ThreadPoolExecutor(max_workers=min(len(frames), 3) or 1) as pool:
            scores = list(pool.map(lambda f: score_frame(f["image"], session), frames))

    per_frame = [
        {"timestamp": f["timestamp"], "generated_probability": None if s is None else round(s, 4)}
        for f, s in zip(frames, scores)
    ]
    valid = [s for s in scores if s is not None]
    if not valid:
        raise RuntimeError("generative detector failed on every frame")
    return round(aggregate(valid), 4), per_frame
