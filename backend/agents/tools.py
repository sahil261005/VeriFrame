import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)

# try to load mediapipe for face landmark tool
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False


def analyze_noise_pattern(frame_data):
    """
    tool: analyze_noise_pattern
    description: Examines the high-frequency noise residual and edge sharpness of a video frame.
    Real camera sensors produce natural noise variance (> 0.05). AI generators produce
    unnaturally smooth noise (< 0.002), which is a strong indicator of synthetic content.
    
    Returns noise variance, laplacian variance (edge sharpness), and a risk assessment.
    """
    img = frame_data["image"]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # compute laplacian variance (measures edge sharpness)
    laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    # compute high-frequency noise residual
    # blur the image and subtract from original to isolate noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    noise = gray.astype(float) - blurred.astype(float)
    noise_var = float(np.var(noise))

    # assess risk level
    risk = "low"
    if noise_var < 0.002:
        risk = "high"  # suspiciously clean noise = likely AI-generated
    elif noise_var < 0.01:
        risk = "medium"

    edge_risk = "normal"
    if laplacian_var < 80.0:
        edge_risk = "blurry (potential face blending)"
    elif laplacian_var > 600.0:
        edge_risk = "oversharpened (potential AI artifact)"

    return {
        "tool": "analyze_noise_pattern",
        "noise_variance": round(noise_var, 6),
        "laplacian_variance": round(laplacian_var, 2),
        "noise_risk": risk,
        "edge_assessment": edge_risk,
        "summary": f"Noise variance: {noise_var:.6f} ({risk} risk). Edge sharpness: {laplacian_var:.2f} ({edge_risk})."
    }


def check_face_landmarks(frame_data):
    """
    tool: check_face_landmarks
    description: Runs MediaPipe Face Mesh on a single frame to detect facial landmarks.
    Returns key landmark positions (nose, eyes, chin) and checks for structural anomalies
    like asymmetric face proportions that indicate face-swap or deepfake artifacts.
    """
    if not MEDIAPIPE_AVAILABLE:
        return {
            "tool": "check_face_landmarks",
            "face_detected": False,
            "summary": "MediaPipe not available on this server."
        }

    img = frame_data["image"]
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]

    face_mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        min_detection_confidence=0.5
    )

    result = face_mesh.process(img_rgb)
    face_mesh.close()

    if not result.multi_face_landmarks:
        return {
            "tool": "check_face_landmarks",
            "face_detected": False,
            "summary": "No face detected in this frame."
        }

    landmarks = result.multi_face_landmarks[0]
    # key landmarks: 1=nose tip, 33=left eye inner, 263=right eye inner, 152=chin
    key_indices = {"nose": 1, "left_eye": 33, "right_eye": 263, "chin": 152}
    points = {}
    for name, idx in key_indices.items():
        lm = landmarks.landmark[idx]
        points[name] = {"x": round(lm.x * w, 1), "y": round(lm.y * h, 1)}

    # check face symmetry: distance from nose to left eye vs nose to right eye
    nose = points["nose"]
    left_dist = abs(nose["x"] - points["left_eye"]["x"])
    right_dist = abs(nose["x"] - points["right_eye"]["x"])

    # asymmetry ratio: perfectly symmetric = 1.0, > 1.5 is suspicious
    if min(left_dist, right_dist) > 0:
        asymmetry = max(left_dist, right_dist) / min(left_dist, right_dist)
    else:
        asymmetry = 1.0

    is_suspicious = asymmetry > 1.5

    return {
        "tool": "check_face_landmarks",
        "face_detected": True,
        "landmarks": points,
        "asymmetry_ratio": round(asymmetry, 3),
        "is_suspicious": is_suspicious,
        "summary": f"Face detected. Asymmetry ratio: {asymmetry:.3f}. {'Suspicious asymmetry detected.' if is_suspicious else 'Face proportions appear normal.'}"
    }


def compare_adjacent_frames(frame_data, all_frames):
    """
    tool: compare_adjacent_frames
    description: Computes dense optical flow between this frame and the next frame in sequence.
    High flow magnitude indicates rapid motion or a temporal glitch. Returns the average
    pixel displacement and whether it exceeds the anomaly threshold.
    """
    timestamp = frame_data["timestamp"]
    curr_img = frame_data["image"]

    # find the next frame in the sequence
    next_frame = None
    for f in all_frames:
        if f["timestamp"] > timestamp:
            next_frame = f
            break

    if next_frame is None:
        return {
            "tool": "compare_adjacent_frames",
            "has_next_frame": False,
            "summary": "No subsequent frame available for comparison."
        }

    prev_gray = cv2.cvtColor(curr_img, cv2.COLOR_BGR2GRAY)
    next_gray = cv2.cvtColor(next_frame["image"], cv2.COLOR_BGR2GRAY)

    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, next_gray, None,
        pyr_scale=0.5, levels=3, winsize=15,
        iterations=3, poly_n=5, poly_sigma=1.2, flags=0
    )

    mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    avg_mag = float(np.mean(mag))
    max_mag = float(np.max(mag))

    # anything above 8.0 average is very unusual for adjacent keyframes
    is_anomalous = avg_mag > 8.0

    return {
        "tool": "compare_adjacent_frames",
        "has_next_frame": True,
        "avg_flow_magnitude": round(avg_mag, 4),
        "max_flow_magnitude": round(max_mag, 4),
        "is_anomalous": is_anomalous,
        "summary": f"Optical flow avg: {avg_mag:.4f}, max: {max_mag:.4f}. {'Anomalous motion detected!' if is_anomalous else 'Motion appears normal.'}"
    }


def check_metadata(metadata):
    """
    tool: check_metadata
    description: Examines the video container metadata and provenance information.
    Checks for C2PA cryptographic signatures, encoder type, whether camera metadata
    was stripped, and if the filename suggests a camera source or social media re-encoding.
    """
    provenance = metadata.get("provenance", {})

    return {
        "tool": "check_metadata",
        "encoder": provenance.get("encoder", "unknown"),
        "c2pa_compliant": provenance.get("c2pa_compliant", False),
        "metadata_stripped": provenance.get("metadata_stripped", True),
        "is_camera_filename": provenance.get("is_camera_filename", False),
        "is_social_filename": provenance.get("is_social_filename", False),
        "provenance_score": provenance.get("provenance_score", 0.5),
        "summary": f"Encoder: {provenance.get('encoder', 'unknown')}. C2PA: {'Yes' if provenance.get('c2pa_compliant') else 'No'}. Metadata stripped: {'Yes' if provenance.get('metadata_stripped', True) else 'No'}."
    }


# registry of all available tools so the LLM agent can look them up by name
TOOL_REGISTRY = {
    "analyze_noise_pattern": {
        "function": analyze_noise_pattern,
        "needs_frames": False,  # only needs the single frame
        "needs_metadata": False,
        "description": "Examines noise residual and edge sharpness to detect AI-generated content."
    },
    "check_face_landmarks": {
        "function": check_face_landmarks,
        "needs_frames": False,
        "needs_metadata": False,
        "description": "Runs face mesh detection and checks facial symmetry for deepfake indicators."
    },
    "compare_adjacent_frames": {
        "function": compare_adjacent_frames,
        "needs_frames": True,   # needs the full frames list to find the next frame
        "needs_metadata": False,
        "description": "Computes optical flow between this frame and the next to detect temporal glitches."
    },
    "check_metadata": {
        "function": check_metadata,
        "needs_frames": False,
        "needs_metadata": True,  # needs video metadata dict
        "description": "Examines video provenance, encoder info, and C2PA compliance."
    }
}
