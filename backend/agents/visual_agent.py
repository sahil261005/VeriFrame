import os
import io
import requests
from PIL import Image
import cv2
import logging
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

# HuggingFace model API endpoints
HF_MODEL_ID = "dima806/deepfake_vs_real_image_detection"
HF_ROUTER_URL = f"https://router.huggingface.co/hf-inference/models/{HF_MODEL_ID}"
HF_LEGACY_URL = f"https://api-inference.huggingface.co/models/{HF_MODEL_ID}"


def load_model():
    """
    tries to load local transformers model.
    if transformers is missing or on cloud server (Render), uses 'hf_api'.
    """
    try:
        from transformers import pipeline
        return pipeline("image-classification", model=HF_MODEL_ID)
    except Exception:
        # On Render or CPU servers, use HuggingFace free API over HTTP
        return "hf_api"


# persistent session for connection pooling & TLS keep-alive
_http_session = requests.Session()


def query_huggingface_api(img_array):
    # sends a frame to HuggingFace cloud API and returns the fake score
    hf_token = os.environ.get("HF_TOKEN")

    # convert OpenCV BGR frame to RGB and PIL Image
    rgb = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)

    buffer = io.BytesIO()
    pil_img.save(buffer, format="JPEG", quality=85)
    image_bytes = buffer.getvalue()

    # Query modern HuggingFace router endpoint with explicit Content-Type
    headers = {
        "Content-Type": "image/jpeg",
        "x-wait-for-model": "true"
    }
    if hf_token:
        headers["Authorization"] = f"Bearer {hf_token}"

    try:
        response = _http_session.post(HF_ROUTER_URL, headers=headers, data=image_bytes, timeout=8)
        if response.status_code == 200:
            predictions = response.json()
            logger.info(f"HuggingFace API response: {predictions}")
            if isinstance(predictions, list):
                for pred in predictions:
                    if isinstance(pred, dict) and "fake" in pred.get("label", "").lower():
                        return float(pred.get("score", 0.0))
        else:
            logger.warning(f"HuggingFace HTTP {response.status_code}: {response.text}")
    except Exception as http_err:
        logger.warning(f"HuggingFace HTTP request to {HF_ROUTER_URL} failed: {http_err}")

    return None


def calculate_heuristic_score(img_array, noise_var):
    """
    fallback function: calculates fake score using simple image noise and edge sharpness
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
    # runs deepfake check on one frame. used by the thread pool below
    img_array = frame_data["image"]
    timestamp = frame_data["timestamp"]
    noise_var = frame_data.get("noise_variance", 0)

    fake_score = None

    # Step 1: Try HuggingFace Cloud API first (for Render server)
    if pipe == "hf_api":
        fake_score = query_huggingface_api(img_array)

    # Step 2: Try local PyTorch model if installed
    elif not isinstance(pipe, str):
        try:
            rgb = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)
            predictions = pipe(pil_img)
            for pred in predictions:
                if "fake" in pred["label"].lower():
                    fake_score = float(pred["score"])
                    break
        except Exception:
            fake_score = None

    # Step 3: Fallback to noise heuristics if API or local model failed
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
    # check all frames for deepfakes. uses threads so we dont wait for each API call one by one
    if not frames:
        return 0.0, [], []

    # run frames through the model in parallel (each one makes an HTTP call so threading helps a lot)
    workers = min(len(frames), 6)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        per_frame_results = list(pool.map(lambda f: analyze_single_frame(f, pipe), frames))

    # Calculate average fake score across all frames
    if len(per_frame_results) > 0:
        total_score = sum(f["fake_confidence"] for f in per_frame_results)
        visual_score = total_score / len(per_frame_results)
    else:
        visual_score = 0.0

    # Get top 5 most suspicious frames
    sorted_frames = sorted(per_frame_results, key=lambda x: x["fake_confidence"], reverse=True)
    flagged = sorted_frames[:5]

    return round(visual_score, 4), flagged, per_frame_results
