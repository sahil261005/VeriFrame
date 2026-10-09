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

# the ViT only knows about face swaps so we use mediapipe to check if a frame even has a face
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    logger.warning("mediapipe not installed, visual agent cannot tell face-less videos apart")

# path to the quantized onnx model, its 83 MB in INT8
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "model_onnx")
ONNX_MODEL_PATH = os.path.join(MODEL_DIR, "model_quantized.onnx")

# by default (and in prod) the face swap ViT runs from the local onnx file in model_onnx/
# if GEN_API_URL is set we send frames to the external service instead so the server doesnt load the model
FACESWAP_BACKEND = os.environ.get("FACESWAP_BACKEND", "remote" if remote_detector.GEN_API_URL else "local")


class _RemoteFaceswap:
    """marker so analyze_frames knows to use the hosted face swap model"""


# onnx session is loaded once and reused for every request
_onnx_session = None
_load_lock = threading.Lock()

# these have to match model_onnx/preprocessor_config.json, not the usual imagenet numbers
_MEAN = np.array([0.5, 0.5, 0.5], dtype=np.float32)
_STD = np.array([0.5, 0.5, 0.5], dtype=np.float32)

# one face detector shared by all requests. mediapipe isnt thread safe so we lock around it
_face_detector = None
_detector_lock = threading.Lock()

# smallest face in pixels (at 480p) that we count as a face
MIN_FACE_SIZE = 24


def load_model():
    """
    loads the quantized onnx model. its the same ViT from huggingface
    but it runs on our server so no API calls needed
    """
    global _onnx_session

    if _onnx_session is not None:
        return _onnx_session

    if FACESWAP_BACKEND == "remote":
        _onnx_session = _RemoteFaceswap()
        logger.info(f"face-swap model will run on the hosted service {remote_detector.GEN_API_URL}")
        return _onnx_session

    # the preload thread at startup and the first request can both get here, so lock it
    with _load_lock:
        if _onnx_session is not None:
            return _onnx_session
        return _load_model_locked()


def _load_model_locked():
    global _onnx_session
    try:
        import onnxruntime as ort

        if os.path.exists(ONNX_MODEL_PATH):
            # prepacking copies the weights again (about 150MB more) and wasnt faster, too much for a 512MB server
            opts = ort.SessionOptions()
            opts.add_session_config_entry("session.disable_prepacking", "1")
            # the cpu arena never gives memory back, turning it off saves about 55 MB
            opts.enable_cpu_mem_arena = False
            opts.enable_mem_pattern = False
            # frames already run in parallel so each run only gets its share of the cpus
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
    true if theres at least one big enough face in the frame.
    we still classify the full frame, cropping to the face made real LFW photos come out ~90% fake
    """
    global _face_detector
    if not MEDIAPIPE_AVAILABLE:
        return True

    h, w = img_array.shape[:2]
    rgb = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)

    with _detector_lock:
        if _face_detector is None:
            # model_selection=1 is the full range model, works for faces up to ~5m away
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
    gets an opencv frame ready for the ViT, resize to 224x224, BGR to RGB and normalize
    """
    rgb = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)

    # the ViT wants 224x224
    pil_img = pil_img.resize((224, 224), Image.BILINEAR)

    # scale pixels to 0-1
    img_np = np.array(pil_img).astype(np.float32) / 255.0

    img_np = (img_np - _MEAN) / _STD

    # HWC to CHW because the model wants channels first
    img_np = np.transpose(img_np, (2, 0, 1))

    # add batch dimension so its (1, 3, 224, 224)
    return np.expand_dims(img_np, axis=0)


def run_onnx_inference(img_array, session):
    """
    runs the ViT on one frame and returns the fake score, 0.0 is real and 1.0 is fake
    """
    try:
        input_tensor = preprocess_frame(img_array)

        input_name = session.get_inputs()[0].name
        output = session.run(None, {input_name: input_tensor})
        logits = output[0][0]  # [real_logit, fake_logit]

        # softmax, subtract the max first so exp doesnt overflow
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / exp_logits.sum()

        # index 1 is the fake class
        fake_score = float(probs[1])
        return fake_score

    except Exception as e:
        logger.warning(f"ONNX inference failed: {e}")
        return None


def calculate_heuristic_score(img_array, noise_var):
    """
    fallback if the onnx model isnt available, guesses a score from noise and edge sharpness
    """
    gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
    laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    base_score = 0.15

    # AI images tend to be too smooth. noise_var uses the same scale as tools.analyze_noise_pattern
    if noise_var < 0.00003:
        base_score += 0.35

    # blurry blending edges or weirdly sharp edges
    if laplacian_var < 80.0:
        base_score += 0.35
    elif laplacian_var > 600.0:
        base_score += 0.15

    # keep it between 0.05 and 0.95
    return min(max(base_score, 0.05), 0.95)


def analyze_single_frame(frame_data, pipe):
    """checks one frame for deepfake stuff"""
    img_array = frame_data["image"]
    timestamp = frame_data["timestamp"]
    noise_var = frame_data.get("noise_variance", 0)

    fake_score = None
    face_found = True

    # first try the onnx ViT, takes about 0.1s on cpu
    if pipe != "heuristic_fallback":
        face_found = frame_has_face(img_array)
        if not face_found:
            # no face means the face swap model cant tell us anything
            return {
                "timestamp": timestamp,
                "fake_confidence": 0.0,
                "noise_variance": noise_var,
                "label": "no_face",
                "face_found": False
            }
        fake_score = run_onnx_inference(img_array, pipe)

    # if the model isnt there or it failed, use the noise heuristic
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
    """gets face swap scores from the hosted model, frames with no face dont get sent"""
    has_face = [frame_has_face(f["image"]) for f in frames]
    idx = [i for i, h in enumerate(has_face) if h]

    remote_scores = {}
    if idx:
        try:
            probs = remote_detector.post_frames("/faceswap", [frames[i] for i in idx])
            remote_scores = dict(zip(idx, probs))
        except Exception as e:
            # if the service is down use the noise heuristic so the whole agent doesnt fail
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
    """checks all the frames for deepfakes in parallel"""
    if not frames:
        return 0.0, [], []

    if isinstance(pipe, _RemoteFaceswap):
        per_frame_results = _analyze_remote(frames)
    else:
        workers = min(len(frames), 6, config.CPU_THREADS)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            per_frame_results = list(pool.map(lambda f: analyze_single_frame(f, pipe), frames))

    # only count frames that actually got classified
    scored = [f for f in per_frame_results if f.get("face_found", True)]

    # average fake score
    if len(scored) > 0:
        total_score = sum(f["fake_confidence"] for f in scored)
        visual_score = total_score / len(scored)
    else:
        visual_score = 0.0

    # top 5 most suspicious frames
    sorted_frames = sorted(scored, key=lambda x: x["fake_confidence"], reverse=True)
    flagged = sorted_frames[:5]

    return round(visual_score, 4), flagged, per_frame_results


def used_heuristics_only(per_frame_results):
    """true if the hosted model was down and every frame used the noise heuristic"""
    scored = [f for f in per_frame_results if f.get("face_found", True)]
    return len(scored) > 0 and all(f.get("source") == "heuristic" for f in scored)


def no_faces_found(per_frame_results):
    """true if the model ran but didnt find a face in any frame"""
    return len(per_frame_results) > 0 and not any(f.get("face_found", True) for f in per_frame_results)
