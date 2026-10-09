import cv2
import numpy as np
import logging
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

# try to load mediapipe, print warning but dont crash if not installed
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    logger.warning("mediapipe not installed, face consistency checks will be skipped")

# each keyframe comes with a "burst" of frames right after it (see preprocessing.extract_frames).
# both checks only compare frames next to each other inside a burst, not keyframes a second apart,
# otherwise normal head or scene movement would look like an artifact

FLOW_WIDTH = 240
SCENE_CUT_CORRELATION = 0.5     # if histogram correlation is below this its a scene cut, not an artifact
STILL_FLOW = 0.05               # below this the frames are basically duplicates
SPIKE_RATIO = 3.0               # a jump that moves 3x more than the usual motion in the burst
SPIKE_MIN_MAGNITUDE = 1.0       # skip spikes that are tiny anyway, in pixels at FLOW_WIDTH
MIN_TRANSITIONS = 4

# face points that barely move when someone talks or smiles, like eye corners, nose, nostrils and forehead
RIGID_LANDMARKS = [33, 133, 362, 263, 168, 6, 197, 195, 5, 4, 1, 98, 327, 10]
LEFT_EYE_OUTER, RIGHT_EYE_OUTER = 33, 263
MIN_EYE_DISTANCE = 20.0         # in pixels, faces smaller than this just give noisy landmarks
MIN_FACE_RUN = 5                # need at least this many frames in a row with a face to measure jitter
JITTER_THRESHOLD = 0.04         # jitter as a fraction of eye distance. real faces measured around 0.01, max 0.025


def _gray_small(img):
    h, w = img.shape[:2]
    small_h = max(1, int(h * (FLOW_WIDTH / w)))
    return cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (FLOW_WIDTH, small_h))


def _is_scene_cut(a, b):
    ha = cv2.calcHist([a], [0], None, [64], [0, 256])
    hb = cv2.calcHist([b], [0], None, [64], [0, 256])
    return cv2.compareHist(ha, hb, cv2.HISTCMP_CORREL) < SCENE_CUT_CORRELATION


def _burst_flow(frame):
    burst = frame.get("burst") or []
    if len(burst) < MIN_TRANSITIONS + 1:
        return None

    grays = [_gray_small(img) for img in burst]
    magnitudes = []
    for prev_gray, curr_gray in zip(grays, grays[1:]):
        if _is_scene_cut(prev_gray, curr_gray):
            continue

        # run farneback to get flow vectors for each pixel
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, curr_gray,
            None,
            pyr_scale=0.5, levels=2, winsize=13,
            iterations=2, poly_n=5, poly_sigma=1.1,
            flags=0
        )
        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        magnitudes.append(float(np.mean(mag)))

    # duplicate frames from fps conversion pull the median to zero and make fake spikes, so drop them
    moving = [m for m in magnitudes if m > STILL_FLOW]
    if len(moving) < MIN_TRANSITIONS:
        return None

    typical = float(np.median(moving))
    peak = max(moving)
    spike_ratio = peak / typical if typical > 0 else 0.0

    return {
        "timestamp": frame["timestamp"],
        "flow_magnitude": round(peak, 4),
        "spike_ratio": round(spike_ratio, 2),
        "is_anomalous": peak > SPIKE_MIN_MAGNITUDE and spike_ratio > SPIKE_RATIO
    }


def compute_optical_flow(frames):
    # look for sudden jumps in motion between frames, like spliced or dropped or glitchy frames
    results = []
    for frame in frames:
        r = _burst_flow(frame)
        if r is not None:
            results.append(r)
    return results


def _similarity_align(src, dst):
    # umeyama alignment, finds the rotation, scale and shift that best maps src onto dst
    mu_s, mu_d = src.mean(axis=0), dst.mean(axis=0)
    s, d = src - mu_s, dst - mu_d
    var_s = (s ** 2).sum() / len(src)
    if var_s == 0:
        return src
    cov = d.T @ s / len(src)
    u, sig, vt = np.linalg.svd(cov)
    sign = np.ones(2)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        sign[-1] = -1
    rot = u @ np.diag(sign) @ vt
    scale = (sig * sign).sum() / var_s
    return (scale * (rot @ s.T)).T + mu_d


def _burst_jitter(face_mesh, frame):
    burst = frame.get("burst") or []
    shapes = []
    eye_dists = []
    for img in burst:
        h, w = img.shape[:2]
        result = face_mesh.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if not result.multi_face_landmarks:
            shapes.append(None)
            continue
        lm = result.multi_face_landmarks[0].landmark
        pts = np.array([(lm[i].x * w, lm[i].y * h) for i in RIGID_LANDMARKS], dtype=np.float64)
        shapes.append(pts)
        eye_dists.append(np.hypot(lm[LEFT_EYE_OUTER].x * w - lm[RIGHT_EYE_OUTER].x * w,
                                  lm[LEFT_EYE_OUTER].y * h - lm[RIGHT_EYE_OUTER].y * h))

    # find the longest run of frames in a row that have a face
    best, run = [], []
    for pts in shapes:
        if pts is None:
            run = []
            continue
        run.append(pts)
        if len(run) > len(best):
            best = list(run)

    if len(best) < MIN_FACE_RUN or not eye_dists:
        return None
    eye_dist = float(np.median(eye_dists))
    if eye_dist < MIN_EYE_DISTANCE:
        return None

    # undo head movement, rotation and zoom so we only compare the face shape
    aligned = [best[0]] + [_similarity_align(p, best[0]) for p in best[1:]]

    # second difference cancels out smooth motion like a slow head turn but jitter still shows up
    jitters = []
    for a, b, c in zip(aligned, aligned[1:], aligned[2:]):
        jitters.append(float(np.mean(np.linalg.norm(a - 2 * b + c, axis=1))) / eye_dist)
    jitter = float(np.median(jitters))

    return {
        "timestamp": frame["timestamp"],
        "landmark_shift": round(jitter, 4),
        "is_inconsistent": jitter > JITTER_THRESHOLD
    }


def check_face_consistency(frames):
    # track the rigid face landmarks across frames.
    # a real face moves smoothly but face swaps usually make the landmarks jitter
    if not MEDIAPIPE_AVAILABLE:
        logger.info("skipping face consistency (mediapipe not available)")
        return []

    # static mode so each frame is detected on its own, tracking mode would smooth the jitter away
    face_mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        min_detection_confidence=0.5
    )
    results = []
    try:
        for frame in frames:
            r = _burst_jitter(face_mesh, frame)
            if r is not None:
                results.append(r)
    finally:
        face_mesh.close()
    return results


def run_temporal_analysis(frames):
    # run both checks at the same time using threads to save time
    with ThreadPoolExecutor(max_workers=2) as pool:
        future_flow = pool.submit(compute_optical_flow, frames)
        future_face = pool.submit(check_face_consistency, frames)
        flow_results = future_flow.result()
        face_results = future_face.result()

    # score each burst. face jitter is pretty specific to face swaps so it counts as 1.0,
    # a motion spike could just be an edit or jump cut so it only counts 0.5
    burst_scores = {}
    for r in flow_results:
        burst_scores[r["timestamp"]] = 0.5 if r["is_anomalous"] else 0.0
    for r in face_results:
        if r["is_inconsistent"]:
            burst_scores[r["timestamp"]] = 1.0
        else:
            burst_scores.setdefault(r["timestamp"], 0.0)

    flagged_timestamps = sorted(t for t, v in burst_scores.items() if v > 0)
    temporal_score = sum(burst_scores.values()) / len(burst_scores) if burst_scores else 0.0

    return round(temporal_score, 4), flagged_timestamps, flow_results, face_results
