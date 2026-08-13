import os
import io
import requests
from PIL import Image
import cv2
import logging

logger = logging.getLogger(__name__)

# HuggingFace model API endpoint for deepfake detection
HF_API_URL = "https://api-inference.huggingface.co/models/dima806/deepfake_vs_real_image_detection"


def load_model():
    """
    tries to load local transformers model.
    if transformers is missing or on cloud server (Render), uses 'hf_api'.
    """
    try:
        from transformers import pipeline
        return pipeline("image-classification", model="dima806/deepfake_vs_real_image_detection")
    except Exception:
        # On Render or CPU servers, use HuggingFace free API over HTTP
        return "hf_api"


def query_huggingface_api(img_array):
    """
    sends a video frame to HuggingFace free API and gets the fake probability score
    """
    try:
        # convert OpenCV BGR frame to RGB and PIL Image
        rgb = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)

        # save PIL image to JPEG bytes
        buffer = io.BytesIO()
        pil_img.save(buffer, format="JPEG")
        image_bytes = buffer.getvalue()

        # add HuggingFace API key and wait-for-model header to prevent 503 cold-start skips
        headers = {
            "x-wait-for-model": "true"
        }
        hf_token = os.environ.get("HF_TOKEN")
        if hf_token:
            headers["Authorization"] = f"Bearer {hf_token}"

        # send POST request to HuggingFace GPU API
        response = requests.post(HF_API_URL, headers=headers, data=image_bytes, timeout=12)
        logger.info(f"HuggingFace Inference API HTTP status: {response.status_code}")

        if response.status_code == 200:
            predictions = response.json()
            logger.info(f"HuggingFace Model Predictions: {predictions}")
            # response format: [{"label": "Fake", "score": 0.88}, {"label": "Real", "score": 0.12}]
            if isinstance(predictions, list):
                for pred in predictions:
                    if isinstance(pred, dict) and "fake" in pred.get("label", "").lower():
                        return float(pred.get("score", 0.0))
        else:
            logger.warning(f"HuggingFace returned status {response.status_code}: {response.text}")
        return None
    except Exception as err:
        logger.warning(f"HuggingFace API request failed: {err}")
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


def analyze_frames(frames, pipe):
    """
    loops over video frames and calculates deepfake confidence score for each frame
    """
    per_frame_results = []

    for frame_data in frames:
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

        per_frame_results.append({
            "timestamp": timestamp,
            "fake_confidence": round(fake_score, 4),
            "noise_variance": noise_var,
            "label": label
        })

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
