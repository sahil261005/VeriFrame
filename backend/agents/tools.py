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
    Real camera sensors produce natural noise variance (> 0.00010). AI generators produce
    unnaturally smooth noise (< 0.00003), which is a strong indicator of synthetic content.
    
    Returns noise variance, laplacian variance (edge sharpness), and a risk assessment.
    """
    img = frame_data["image"]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # compute laplacian variance (measures edge sharpness)
    laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    # compute normalized high-frequency noise residual (0.0 to 1.0 scale)
    norm_gray = gray.astype(np.float32) / 255.0
    blurred = cv2.GaussianBlur(norm_gray, (5, 5), 0)
    noise = norm_gray - blurred
    noise_var = float(np.var(noise))

    # assess risk level based on normalized noise variance
    if noise_var < 0.00003:
        risk = "high (unnaturally smooth noise, characteristic of AI generation)"
    elif noise_var < 0.00008:
        risk = "medium (low sensor noise)"
    else:
        risk = "low (natural camera sensor noise pattern)"

    edge_risk = "normal"
    if laplacian_var < 60.0:
        edge_risk = "soft/blurry (potential face blending boundary)"
    elif laplacian_var > 700.0:
        edge_risk = "oversharpened (potential AI artifact)"

    return {
        "tool": "analyze_noise_pattern",
        "noise_variance": round(noise_var, 8),
        "laplacian_variance": round(laplacian_var, 2),
        "noise_risk": risk,
        "edge_assessment": edge_risk,
        "summary": f"Normalized noise variance: {noise_var:.7f} ({risk}). Edge sharpness: {laplacian_var:.1f} ({edge_risk})."
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
    h, w = img.shape[:2]
    
    # downscale for fast face detection
    small_w = 240
    small_h = int(h * (small_w / w))
    small_img = cv2.resize(img, (small_w, small_h))
    img_rgb = cv2.cvtColor(small_img, cv2.COLOR_BGR2RGB)

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

    # downscale to 240px width for fast optical flow
    h, w = curr_img.shape[:2]
    small_w = 240
    small_h = max(1, int(h * (small_w / w)))
    
    prev_gray = cv2.resize(cv2.cvtColor(curr_img, cv2.COLOR_BGR2GRAY), (small_w, small_h))
    next_gray = cv2.resize(cv2.cvtColor(next_frame["image"], cv2.COLOR_BGR2GRAY), (small_w, small_h))

    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, next_gray, None,
        pyr_scale=0.5, levels=2, winsize=13,
        iterations=2, poly_n=5, poly_sigma=1.1, flags=0
    )

    mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    avg_mag = float(np.mean(mag))
    std_mag = float(np.std(mag))

    time_delta = max(0.1, next_frame["timestamp"] - timestamp)
    norm_mag = avg_mag / time_delta
    is_anomalous = (std_mag > 8.0) or (norm_mag > 25.0)

    return {
        "tool": "compare_adjacent_frames",
        "has_next_frame": True,
        "avg_flow_magnitude": round(avg_mag, 2),
        "flow_std": round(std_mag, 2),
        "is_anomalous": is_anomalous,
        "summary": f"Optical flow (delta={time_delta:.2f}s): avg={avg_mag:.2f}, std={std_mag:.2f}. {'Localized warping or jump detected.' if is_anomalous else 'Smooth motion transition.'}"
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
