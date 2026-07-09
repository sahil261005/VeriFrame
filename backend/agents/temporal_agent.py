import cv2
import numpy as np
import math
import logging

logger = logging.getLogger(__name__)

# try to load mediapipe, print warning but dont crash if not installed
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    logger.warning("mediapipe not installed, face consistency checks will be skipped")


def compute_optical_flow(frames):
    # check how much pixels are moving between frames to find glitches
    if len(frames) < 2:
        return []

    results = []

    prev_gray = cv2.cvtColor(frames[0]["image"], cv2.COLOR_BGR2GRAY)

    for i in range(1, len(frames)):
        curr_gray = cv2.cvtColor(frames[i]["image"], cv2.COLOR_BGR2GRAY)

        # run farneback to get flow vectors for each pixel
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, curr_gray,
            None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2,
            flags=0
        )

        # compute flow magnitude
        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        avg_magnitude = float(np.mean(mag))

        results.append({
            "timestamp": frames[i]["timestamp"],
            "flow_magnitude": round(avg_magnitude, 4),
            "is_anomalous": False  # flag this later in loop
        })

        prev_gray = curr_gray

    # flag values higher than 2 standard deviations from average
    if len(results) > 0:
        magnitudes = [r["flow_magnitude"] for r in results]
        mean_mag = sum(magnitudes) / len(magnitudes)

        # calculate standard deviation manually
        squared_diffs = 0
        for m in magnitudes:
            squared_diffs += (m - mean_mag) ** 2
        std_mag = math.sqrt(squared_diffs / len(magnitudes)) if len(magnitudes) > 0 else 0

        threshold = mean_mag + (2 * std_mag)

        for r in results:
            if r["flow_magnitude"] > threshold:
                r["is_anomalous"] = True

    return results


def check_face_consistency(frames):
    # track facial landmarks over time.
    # if landmarks jump around too much, it means the face mesh is glitching
    if not MEDIAPIPE_AVAILABLE:
        logger.info("skipping face consistency (mediapipe not available)")
        return []

    if len(frames) < 2:
        return []

    face_mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        min_detection_confidence=0.5
    )

    # key landmarks: 1=nose, 33=left eye, 263=right eye, 152=chin
    key_points = [1, 33, 263, 152]

    # get landmarks for all frames first
    frame_landmarks = []
    for frame_data in frames:
        img_rgb = cv2.cvtColor(frame_data["image"], cv2.COLOR_BGR2RGB)
        result = face_mesh.process(img_rgb)

        if result.multi_face_landmarks and len(result.multi_face_landmarks) > 0:
            landmarks = result.multi_face_landmarks[0]
            h, w = frame_data["image"].shape[:2]

            points = {}
            for idx in key_points:
                lm = landmarks.landmark[idx]
                # convert normalized scale to pixels
                points[idx] = (lm.x * w, lm.y * h)

            frame_landmarks.append({
                "timestamp": frame_data["timestamp"],
                "points": points,
                "face_found": True
            })
        else:
            frame_landmarks.append({
                "timestamp": frame_data["timestamp"],
                "points": {},
                "face_found": False
            })

    face_mesh.close()

    # count how many frames had a face
    faces_found = 0
    for fl in frame_landmarks:
        if fl["face_found"]:
            faces_found += 1

    if faces_found < 2:
        logger.info(f"only found faces in {faces_found} frames, not enough for consistency check")
        return []

    # compare face landmarks between consecutive frames
    results = []
    # compare adjacent frames
    for i in range(1, len(frame_landmarks)):
        prev = frame_landmarks[i - 1]
        curr = frame_landmarks[i]

        if not prev["face_found"] or not curr["face_found"]:
            continue

        # get total movement of landmarks
        total_shift = 0
        num_points = 0
        for idx in key_points:
            if idx in prev["points"] and idx in curr["points"]:
                px, py = prev["points"][idx]
                cx, cy = curr["points"][idx]
                dist = math.sqrt((cx - px) ** 2 + (cy - py) ** 2)
                total_shift += dist
                num_points += 1

        avg_shift = total_shift / num_points if num_points > 0 else 0

        # flag if landmark movement is more than 15 pixels (means face shifted too fast)
        is_inconsistent = avg_shift > 15.0

        results.append({
            "timestamp": curr["timestamp"],
            "landmark_shift": round(avg_shift, 2),
            "is_inconsistent": is_inconsistent
        })

    return results


def run_temporal_analysis(frames):
    # run optical flow and face meshes, then merge results
    logger.info("running optical flow analysis...")
    flow_results = compute_optical_flow(frames)

    logger.info("running face consistency analysis...")
    face_results = check_face_consistency(frames)

    # combine timestamps flagged by either test
    flagged_timestamps = []
    for r in flow_results:
        if r["is_anomalous"]:
            flagged_timestamps.append(r["timestamp"])

    for r in face_results:
        if r["is_inconsistent"]:
            if r["timestamp"] not in flagged_timestamps:
                flagged_timestamps.append(r["timestamp"])

    # score is fraction of flagged transitions
    total_transitions = max(len(flow_results), 1)
    temporal_score = len(flagged_timestamps) / total_transitions

    return round(temporal_score, 4), flagged_timestamps, flow_results, face_results

