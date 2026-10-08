import os
import numpy as np
from PIL import Image
import cv2
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
import agents.remote_detector as remote_detector
import config

logger = logging.getLogger(__name__)

# the ViT is a face-swap classifier; mediapipe tells us whether a frame has a face for it to judge
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    logger.warning("mediapipe not installed, visual agent cannot tell face-less videos apart")

# path to the quantized ONNX deepfake detection model (83 MB, INT8)
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "model_onnx")
ONNX_MODEL_PATH = os.path.join(MODEL_DIR, "model_quantized.onnx")

# where the face-swap ViT runs. by default (and in production) the local ONNX file in model_onnx/ is used;
# setting GEN_API_URL sends frames to an external scoring service instead, so the host doesn't hold the model.
FACESWAP_BACKEND = os.environ.get("FACESWAP_BACKEND", "remote" if remote_detector.GEN_API_URL else "local")


class _RemoteFaceswap:
    """marker for the hosted face-swap model; see analyze_frames."""


# global ONNX session (loaded once, reused for all requests)
_onnx_session = None
_load_lock = threading.Lock()

# must match model_onnx/preprocessor_config.json (ViTImageProcessor), not ImageNet stats
_MEAN = np.array([0.5, 0.5, 0.5], dtype=np.float32)
_STD = np.array([0.5, 0.5, 0.5], dtype=np.float32)

# face detector is shared across requests; mediapipe graphs are not thread-safe, so calls are serialized
_face_detector = None
_detector_lock = threading.Lock()

# smallest face (pixels, at the 480p analysis resolution) that counts as present
MIN_FACE_SIZE = 24


def load_model():
    """
    loads the quantized ONNX deepfake detection model into memory.
    this runs the same ViT neural network that was on HuggingFace,
    but now runs directly on the server with zero external API calls.
    """
    global _onnx_session

    # if already loaded, return the cached session
    if _onnx_session is not None:
        return _onnx_session

    if FACESWAP_BACKEND == "remote":
        _onnx_session = _RemoteFaceswap()
        logger.info(f"face-swap model will run on the hosted service {remote_detector.GEN_API_URL}")
        return _onnx_session

    # startup preload thread and the first request can race; load only once
    with _load_lock:
        if _onnx_session is not None:
            return _onnx_session
        return _load_model_locked()


def _load_model_locked():
    global _onnx_session
    try:
        import onnxruntime as ort

        if os.path.exists(ONNX_MODEL_PATH):
            # prepacking duplicates the weights in memory (+~150MB, no speed gain here); matters on a 512MB host
            opts = ort.SessionOptions()
            opts.add_session_config_entry("session.disable_prepacking", "1")
            # the CPU arena keeps every buffer it ever allocated; without it inference memory is returned (-55 MB)
            opts.enable_cpu_mem_arena = False
            opts.enable_mem_pattern = False
            # frames already run in parallel (analyze_video), so each run gets its share of the allowed CPUs
            opts.intra_op_num_threads = max(1, config.CPU_THREADS // min(6, config.CPU_THREADS))
            opts.inter_op_num_threads = 1
            _onnx_session = ort.InferenceSession(ONNX_MODEL_PATH, opts)
            logger.info(f"ONNX deepfake model loaded successfully ({os.path.getsize(ONNX_MODEL_PATH) / (1024*1024):.1f} MB)")
            return _onnx_session
        else:
            logger.warning(f"ONNX model file not found at {ONNX_MODEL_PATH}, using heuristic fallback")
            return "heuristic_fallback"
    except Exception as e:
        logger.warning(f"Failed to load ONNX model: {e}, using heuristic fallback")
        return "heuristic_fallback"


def frame_has_face(img_array):
    """
    true if the frame contains at least one face large enough to matter.
    the classifier still runs on the full frame: tight face crops were measured to push
    real faces (LFW photos) to ~90% "fake", so cropping was not adopted.
    """
    global _face_detector
    if not MEDIAPIPE_AVAILABLE:
        return True

    h, w = img_array.shape[:2]
    rgb = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)

    with _detector_lock:
        if _face_detector is None:
            # model_selection=1 is the full-range model (faces up to ~5m away)
            _face_detector = mp.solutions.face_detection.FaceDetection(
                model_selection=1, min_detection_confidence=0.5
            )
        result = _face_detector.process(rgb)

    if not result.detections:
        return False

    for det in result.detections:
        box = det.location_data.relative_bounding_box
        if min(box.width * w, box.height * h) >= MIN_FACE_SIZE:
            return True
    return False


def preprocess_frame(img_array):
    """
    prepares an OpenCV BGR frame for the ViT model.
    resizes to 224x224, converts to RGB, normalizes pixel values.
    """
    # convert BGR to RGB
    rgb = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)

    # resize to 224x224 (what the ViT model expects)
    pil_img = pil_img.resize((224, 224), Image.BILINEAR)

    # convert to numpy array and normalize to [0, 1]
    img_np = np.array(pil_img).astype(np.float32) / 255.0

    img_np = (img_np - _MEAN) / _STD

    # convert from HWC (height, width, channels) to CHW (channels, height, width)
    img_np = np.transpose(img_np, (2, 0, 1))

    # add batch dimension: (1, 3, 224, 224)
    return np.expand_dims(img_np, axis=0)


def run_onnx_inference(img_array, session):
    """
    runs the ONNX ViT deepfake classifier on a single frame.
    returns the fake probability score (0.0 = real, 1.0 = fake).
    """
    try:
        # preprocess the frame for the ViT model
        input_tensor = preprocess_frame(img_array)

        # run inference
        input_name = session.get_inputs()[0].name
        output = session.run(None, {input_name: input_tensor})
        logits = output[0][0]  # shape: (2,) -> [real_logit, fake_logit]

        # softmax to convert logits to probabilities
        exp_logits = np.exp(logits - np.max(logits))  # numerically stable softmax
        probs = exp_logits / exp_logits.sum()

        # index 1 = "Fake" probability
        fake_score = float(probs[1])
        return fake_score

    except Exception as e:
        logger.warning(f"ONNX inference failed: {e}")
        return None


def calculate_heuristic_score(img_array, noise_var):
    """
    fallback function: calculates fake score using simple image noise and edge sharpness.
    used only when the ONNX model is unavailable.
    """
    gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
    laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    base_score = 0.15

    # AI images are unnaturally smooth (normalized noise variance, same scale as tools.analyze_noise_pattern)
    if noise_var < 0.00003:
        base_score += 0.35

    # check for blurry face blending edges or artificial over-sharpness
    if laplacian_var < 80.0:
        base_score += 0.35
    elif laplacian_var > 600.0:
        base_score += 0.15

    # cap final score between 0.05 and 0.95
    return min(max(base_score, 0.05), 0.95)


def analyze_single_frame(frame_data, pipe):
    """runs deepfake detection on one video frame."""
    img_array = frame_data["image"]
    timestamp = frame_data["timestamp"]
    noise_var = frame_data.get("noise_variance", 0)

    fake_score = None
    face_found = True

    # Step 1: Try ONNX ViT deep learning model (runs on server CPU in ~0.1s)
    if pipe != "heuristic_fallback":
        face_found = frame_has_face(img_array)
        if not face_found:
            # a face-swap classifier says nothing about frames without a face
            return {
                "timestamp": timestamp,
                "fake_confidence": 0.0,
                "noise_variance": noise_var,
                "label": "no_face",
                "face_found": False
            }
        fake_score = run_onnx_inference(img_array, pipe)

    # Step 2: Fallback to noise heuristics if ONNX model unavailable or failed
    if fake_score is None:
        fake_score = calculate_heuristic_score(img_array, noise_var)

    label = "fake" if fake_score > 0.5 else "real"

    return {
        "timestamp": timestamp,
        "fake_confidence": round(fake_score, 4),
        "noise_variance": noise_var,
        "label": label,
        "face_found": True
    }


def _analyze_remote(frames):
    """face-swap scores from the hosted model; frames without a face are never sent."""
    has_face = [frame_has_face(f["image"]) for f in frames]
    idx = [i for i, h in enumerate(has_face) if h]

    remote_scores = {}
    if idx:
        try:
            probs = remote_detector.post_frames("/faceswap", [frames[i] for i in idx])
            remote_scores = dict(zip(idx, probs))
        except Exception as e:
            # service unreachable: degrade to the noise heuristics instead of failing the whole agent
            logger.warning(f"hosted face-swap model unavailable, using heuristics: {e}")

    results = []
    for i, f in enumerate(frames):
        base = {"timestamp": f["timestamp"], "noise_variance": f.get("noise_variance", 0)}
        if not has_face[i]:
            results.append({**base, "fake_confidence": 0.0, "label": "no_face", "face_found": False})
            continue
        if i in remote_scores:
            score, source = remote_scores[i], "model"
        else:
            score, source = calculate_heuristic_score(f["image"], f.get("noise_variance", 0)), "heuristic"
        results.append({**base, "fake_confidence": round(score, 4), "label": "fake" if score > 0.5 else "real",
                        "face_found": True, "source": source})
    return results


def analyze_frames(frames, pipe):
    """checks all extracted video frames for deepfakes in parallel."""
    if not frames:
        return 0.0, [], []

    if isinstance(pipe, _RemoteFaceswap):
        per_frame_results = _analyze_remote(frames)
    else:
        workers = min(len(frames), 6, config.CPU_THREADS)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            per_frame_results = list(pool.map(lambda f: analyze_single_frame(f, pipe), frames))

    # only frames that were actually classified count toward the score
    scored = [f for f in per_frame_results if f.get("face_found", True)]

    # calculate average fake score across all scored frames
    if len(scored) > 0:
        total_score = sum(f["fake_confidence"] for f in scored)
        visual_score = total_score / len(scored)
    else:
        visual_score = 0.0

    # get top 5 most suspicious frames
    sorted_frames = sorted(scored, key=lambda x: x["fake_confidence"], reverse=True)
    flagged = sorted_frames[:5]

    return round(visual_score, 4), flagged, per_frame_results


def used_heuristics_only(per_frame_results):
    """true when the hosted model was unreachable and every scored frame fell back to noise heuristics."""
    scored = [f for f in per_frame_results if f.get("face_found", True)]
    return len(scored) > 0 and all(f.get("source") == "heuristic" for f in scored)


def no_faces_found(per_frame_results):
    """true when the model ran but no frame contained a face it could classify."""
    return len(per_frame_results) > 0 and not any(f.get("face_found", True) for f in per_frame_results)
