import os
# Configure HuggingFace client to fail fast if blocked by proxies
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["HF_HUB_MAX_RETRIES"] = "1"
os.environ["HF_HUB_ETAG_TIMEOUT"] = "1"

from PIL import Image
import numpy as np
import cv2
import logging

logger = logging.getLogger(__name__)

# load model once and cache it here so we dont do it every run
_pipeline = None

def load_model():
    global _pipeline
    if _pipeline is not None:
        return _pipeline

    try:
        # import inside so it doesnt crash if transformers package is missing
        from transformers import pipeline
        logger.info("loading deepfake detection model from huggingface... (first time takes a while)")
        _pipeline = pipeline(
            "image-classification",
            model="dima806/deepfake_vs_real_image_detection"
        )
        logger.info("model loaded successfully.")
    except Exception as e:
        logger.warning(f"could not load huggingface model ({e}). using fallback visual heuristics.")
        _pipeline = "fallback"

    return _pipeline


def analyze_frames(frames, pipe):
    # takes frame list from extraction and runs them through the classifier
    per_frame_results = []

    for frame_data in frames:
        img_array = frame_data["image"]
        timestamp = frame_data["timestamp"]
        noise_var = frame_data.get("noise_variance", 0)

        if pipe == "fallback":
            gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
            laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

            # Heuristics: AI-generated images lack natural camera high-frequency sensor noise.
            # Normal camera noise variance is typically > 0.05. Very smooth noise (< 0.005) is highly suspicious.
            base_fake = 0.15
            if noise_var < 0.002:
                base_fake += 0.35  # Suspiciously smooth/clean noise
            if laplacian_var < 80.0:
                base_fake += 0.35  # Blurry face / blending outlines
            elif laplacian_var > 600.0:
                base_fake += 0.15  # Extremely sharp artificial edges (common in AI generations)

            fake_score = min(max(base_fake, 0.05), 0.95)
            label = "fake" if fake_score > 0.5 else "real"
        else:
            # convert BGR to RGB and then to PIL image
            rgb = img_array[:, :, ::-1]
            pil_img = Image.fromarray(rgb)

            # run model
            try:
                predictions = pipe(pil_img)
                model_score = 0.0
                for pred in predictions:
                    if "fake" in pred["label"].lower():
                        model_score = pred["score"]
                        break

                # add a penalty if the frame is too smooth (indicates AI generation)
                noise_penalty = 0.0
                if noise_var < 0.002:
                    noise_penalty = 0.40

                # combine model score and noise check
                fake_score = model_score + noise_penalty
                
                # cap it between 0.05 and 0.95
                if fake_score > 0.95:
                    fake_score = 0.95
                elif fake_score < 0.05:
                    fake_score = 0.05

                label = "fake" if fake_score > 0.5 else "real"
            except Exception as e:
                logger.warning(f"model failed on frame at {timestamp}s: {e}")
                fake_score = 0.0
                label = "error"

        per_frame_results.append({
            "timestamp": timestamp,
            "fake_confidence": round(fake_score, 4),
            "noise_variance": noise_var,
            "label": label
        })

    # avg fake score of all frames
    if len(per_frame_results) > 0:
        total = 0
        for r in per_frame_results:
            total += r["fake_confidence"]
        visual_score = total / len(per_frame_results)
    else:
        visual_score = 0.0

    # get top 5 most suspicious frames
    sorted_frames = sorted(per_frame_results, key=lambda x: x["fake_confidence"], reverse=True)
    flagged = sorted_frames[:5]

    return round(visual_score, 4), flagged, per_frame_results

