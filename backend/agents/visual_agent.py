import os
import numpy as np
from PIL import Image
import cv2
import logging
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

# path to the quantized ONNX deepfake detection model (83 MB, INT8)
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "model_onnx")
ONNX_MODEL_PATH = os.path.join(MODEL_DIR, "model_quantized.onnx")

# global ONNX session (loaded once, reused for all requests)
_onnx_session = None


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

    try:
        import onnxruntime as ort

        if os.path.exists(ONNX_MODEL_PATH):
            _onnx_session = ort.InferenceSession(ONNX_MODEL_PATH)
            logger.info(f"ONNX deepfake model loaded successfully ({os.path.getsize(ONNX_MODEL_PATH) / (1024*1024):.1f} MB)")
            return _onnx_session
        else:
            logger.warning(f"ONNX model file not found at {ONNX_MODEL_PATH}, using heuristic fallback")
            return "heuristic_fallback"
    except Exception as e:
        logger.warning(f"Failed to load ONNX model: {e}, using heuristic fallback")
        return "heuristic_fallback"


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

    # apply ImageNet normalization (mean and std from training data)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_np = (img_np - mean) / std

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

    # AI images are unnaturally smooth (very low noise variance)
    if noise_var < 0.002:
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

    # Step 1: Try ONNX ViT deep learning model (runs on server CPU in ~0.1s)
    if pipe != "heuristic_fallback":
        fake_score = run_onnx_inference(img_array, pipe)

    # Step 2: Fallback to noise heuristics if ONNX model unavailable or failed
    if fake_score is None:
        fake_score = calculate_heuristic_score(img_array, noise_var)

    label = "fake" if fake_score > 0.5 else "real"

    return {
        "timestamp": timestamp,
        "fake_confidence": round(fake_score, 4),
        "noise_variance": noise_var,
        "label": label
    }


def analyze_frames(frames, pipe):
    """checks all extracted video frames for deepfakes in parallel."""
    if not frames:
        return 0.0, [], []

    workers = min(len(frames), 6)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        per_frame_results = list(pool.map(lambda f: analyze_single_frame(f, pipe), frames))

    # calculate average fake score across all frames
    if len(per_frame_results) > 0:
        total_score = sum(f["fake_confidence"] for f in per_frame_results)
        visual_score = total_score / len(per_frame_results)
    else:
        visual_score = 0.0

    # get top 5 most suspicious frames
    sorted_frames = sorted(per_frame_results, key=lambda x: x["fake_confidence"], reverse=True)
    flagged = sorted_frames[:5]

    return round(visual_score, 4), flagged, per_frame_results
