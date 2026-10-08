import os
import subprocess
import shutil
import re
import json
import logging
import numpy as np
import cv2

import config

cv2.setNumThreads(config.CPU_THREADS)  # OpenCV's own pool also defaults to every visible core

logger = logging.getLogger(__name__)


# real cameras have sensor noise but AI generated frames are usually super clean and smooth
# normalized 0-1 scale, identical to agents.tools.analyze_noise_pattern so all thresholds agree
def compute_noise_residual(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    return float(np.var(gray - blurred))


# IPTC digital source types (the vocabulary C2PA uses) that declare media was made by generative AI
_AI_SOURCE_TYPES = ("trainedalgorithmicmedia", "compositewithtrainedalgorithmicmedia")


def read_c2pa(file_path):
    """
    reads C2PA Content Credentials embedded in the video, if any. AI tools that support C2PA record a
    digitalSourceType of "trainedAlgorithmicMedia" when they generate content; edits keep the original
    manifest as an ingredient, so every manifest in the store is checked, not just the latest one.
    """
    result = {
        "c2pa_present": False,
        "c2pa_signature_valid": False,
        "c2pa_trusted_signer": False,
        "c2pa_ai_generated": False,
        "c2pa_generator": None,
        "c2pa_source_types": [],
    }
    try:
        import c2pa  # c2pa-python; optional (needs python >= 3.10)
    except ImportError:
        logger.info("c2pa-python not installed; skipping Content Credentials check")
        return result

    try:
        reader = c2pa.Reader.try_create(file_path)
        if reader is None:
            return result
        with reader:
            store = json.loads(reader.json())
            state = str(reader.get_validation_state() or "")
    except Exception as e:
        logger.warning(f"could not read C2PA data: {e}")
        return result

    manifests = store.get("manifests", {}) or {}
    active = manifests.get(store.get("active_manifest"), {}) or {}
    generator = active.get("claim_generator")
    if not generator and active.get("claim_generator_info"):
        generator = active["claim_generator_info"][0].get("name")

    source_types = set()
    for manifest in manifests.values():
        for assertion in manifest.get("assertions", []) or []:
            if not str(assertion.get("label", "")).startswith("c2pa.actions"):
                continue
            for action in (assertion.get("data", {}) or {}).get("actions", []) or []:
                if action.get("digitalSourceType"):
                    source_types.add(action["digitalSourceType"])

    result.update({
        "c2pa_present": True,
        # "Valid" = signature and hashes check out; "Trusted" additionally = signer is on a trust list
        "c2pa_signature_valid": state.lower() in ("valid", "trusted"),
        "c2pa_trusted_signer": state.lower() == "trusted",
        "c2pa_ai_generated": any(t.rstrip("/").split("/")[-1].lower() in _AI_SOURCE_TYPES for t in source_types),
        "c2pa_generator": generator,
        "c2pa_source_types": sorted(source_types),
    })
    return result


def check_provenance(file_path):
    # set up default values
    provenance_score = 0.5
    c2pa_compliant = False
    encoder = "unknown"
    metadata_stripped = True
    is_camera_filename = False
    is_social_filename = False

    # get filename in lowercase to check patterns
    filename = os.path.basename(file_path).lower()

    # check if it looks like a camera name
    if filename.startswith("dji_") or filename.startswith("gopr") or filename.startswith("pxl_") or filename.startswith("img_") or filename.startswith("vid_") or filename.startswith("dscf"):
        is_camera_filename = True
        provenance_score = 0.85
    # check if it looks like a social media filename
    elif "whatsapp" in filename or "tiktok" in filename or "snapchat" in filename or "instagram" in filename or "telegram" in filename or "facebook" in filename:
        is_social_filename = True
        provenance_score = 0.70

    # run ffprobe to see if we can find metadata
    ffprobe_path = shutil.which("ffprobe")
    if ffprobe_path:
        try:
            cmd = ["ffprobe", "-v", "quiet", "-show_format", "-show_streams", file_path]
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=30)
            output = result.stdout.lower()

            # if we find encoder in output, it means metadata is there
            if "encoder" in output:
                metadata_stripped = False
                # find the encoder line and extract name
                for line in output.split("\n"):
                    if "encoder=" in line:
                        parts = line.split("encoder=")
                        if len(parts) > 1:
                            encoder = parts[1].strip()
                        break

        except Exception as e:
            logger.error(f"Error running ffprobe: {e}")
    else:
        logger.warning("ffprobe not found, skipping container checks")

    # real Content Credentials check (replaces the old keyword search in ffprobe output)
    c2pa_info = read_c2pa(file_path)
    if c2pa_info["c2pa_present"]:
        metadata_stripped = False
        c2pa_compliant = c2pa_info["c2pa_signature_valid"]
        if c2pa_info["c2pa_ai_generated"]:
            provenance_score = 0.02
        elif c2pa_compliant:
            provenance_score = 0.98

    # if metadata is gone and it is not a camera/social name, it is low trust
    if metadata_stripped:
        if not is_camera_filename and not is_social_filename:
            provenance_score = 0.30

    return {
        "provenance_score": provenance_score,
        "c2pa_compliant": c2pa_compliant,
        "encoder": encoder,
        "metadata_stripped": metadata_stripped,
        "is_camera_filename": is_camera_filename,
        "is_social_filename": is_social_filename,
        **c2pa_info
    }


# get basic metadata from opencv. we need this to validate and show on frontend later
def get_video_metadata(file_path):
    cap = cv2.VideoCapture(file_path)
    if not cap.isOpened():
        raise ValueError("could not open video file to read metadata")
    
    # query the properties we need
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # convert the binary fourcc code to a readable string
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    codec = ""
    for i in range(4):
        codec += chr((fourcc >> (8 * i)) & 0xFF)
        
    duration = 0.0
    if fps > 0:
        duration = total_frames / fps
        
    cap.release()
    
    # opencv doesnt give bitrate directly so we do it by hand
    bitrate = 0
    try:
        file_size = os.path.getsize(file_path)
        if duration > 0:
            # file size in bits / duration in seconds
            bitrate = int((file_size * 8) / duration)
    except Exception:
        # just set it to 0 if it fails
        pass
        
    # run our metadata checks
    provenance = check_provenance(file_path)
    
    # calculate adversarial robustness score based on resolution and bitrate thresholds
    # low-resolution and low-bitrate streams make it easy to hide manipulation signatures
    robustness = 1.0
    
    if width < 1280 or height < 720:
        robustness -= 0.20
    if width < 640 or height < 480:
        robustness -= 0.30
        
    if bitrate > 0:
        if bitrate < 500000:  # < 500 kbps
            robustness -= 0.15
        if bitrate < 200000:  # < 200 kbps
            robustness -= 0.25
            
    robustness_score = round(max(0.1, robustness), 2)
        
    return {
        "fps": round(fps, 2),
        "duration": round(duration, 2),
        "width": width,
        "height": height,
        "codec": codec,
        "total_frames": total_frames,
        "bitrate": bitrate,
        "robustness_score": robustness_score,
        "provenance": provenance
    }

# basic validation. we check if it exists, format is ok, and length is under 30s.
# we dont want people uploading massive movies and crashing the server
def validate_video(file_path):
    if not os.path.exists(file_path):
        raise ValueError("video file does not exist")
        
    # check extension is supported
    ext = os.path.splitext(file_path)[1].lower().replace(".", "")
    allowed = ["mp4", "avi", "mov", "webm"]
    if ext not in allowed:
        raise ValueError(f"unsupported format: {ext}. we only support mp4, avi, mov, webm")
        
    # get metadata and check duration
    meta = get_video_metadata(file_path)
    if meta["duration"] > 30.0:
        raise ValueError(f"video duration is {meta['duration']}s, which exceeds the 30-second limit")

    # big videos are accepted and shrunk to 480p before analysis (see downscale_for_analysis), because decoding
    # full-size frames all at once needs hundreds of MB. the hard ceiling only stops absurd inputs (8K and up).
    max_input_pixels = int(os.environ.get("MAX_INPUT_PIXELS", "9000000"))
    max_pixels = int(os.environ.get("MAX_VIDEO_PIXELS", "1000000"))
    w, h = meta.get("width", 0) or 0, meta.get("height", 0) or 0
    if w * h > max_input_pixels:
        raise ValueError(f"video resolution is too high ({w}x{h}); please upload 4K or lower")
    meta["needs_downscale"] = w * h > max_pixels

    return meta

# downscales to 480p so we save bandwidth and processing power.
# keeps aspect ratio same
def downscale_video(file_path, output_path, target_height=480):
    cap = cv2.VideoCapture(file_path)
    if not cap.isOpened():
        raise ValueError("could not open video to downscale")
        
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # dont upscale it if its already smaller than 480p
    if height <= target_height:
        # write with original size
        new_height = height
        new_width = width
    else:
        # compute width based on height
        aspect = width / height
        new_height = target_height
        new_width = int(target_height * aspect)
        
        # make sure width is even or encoders will crash
        if new_width % 2 != 0:
            new_width += 1
            
    # mp4v works everywhere so we use it
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (new_width, new_height))
    
    # loop frames, resize them, and write
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        resized = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
        out.write(resized)
        
    cap.release()
    out.release()
    return output_path


# shrinks a large upload to 480p for the visual/temporal/LLM agents. video only: audio forensics still reads the
# original file. uses ffmpeg when installed (fast, low memory) and otherwise the frame-by-frame opencv path above,
# which holds a single frame in memory at a time.
def downscale_for_analysis(file_path, output_path, target_height=480, timeout=180):
    if shutil.which("ffmpeg"):
        cmd = ["ffmpeg", "-y", "-i", file_path, "-an", "-vf", f"scale=-2:{target_height}",
               "-c:v", "libx264", "-preset", "ultrafast", "-crf", "26", "-threads", "2", output_path]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
            if result.returncode == 0 and os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                return output_path
            logger.warning("ffmpeg downscale failed, falling back to opencv: %s", result.stderr[-300:])
        except Exception as e:
            logger.warning("ffmpeg downscale errored, falling back to opencv: %s", e)
    return downscale_video(file_path, output_path, target_height=target_height)

# number of consecutive frames captured after each keyframe for the temporal agent,
# and the width they are stored at (small, to keep memory low)
BURST_LENGTH = 8
BURST_WIDTH = 480


def _resize_to_width(frame, width):
    h, w = frame.shape[:2]
    if w <= width:
        return frame
    return cv2.resize(frame, (width, max(1, int(h * width / w))), interpolation=cv2.INTER_AREA)


# pull sample frames using fast sequential grab (avoids slow container seek stalls)
def extract_frames(file_path, interval=1.0, target_height=480, max_frames=6):
    cap = cv2.VideoCapture(file_path)
    if not cap.isOpened():
        raise ValueError("could not open video to extract frames")
        
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0
        
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 10.0
    
    # pick 5-6 evenly spaced keyframes across the whole video
    num_samples = min(max_frames, max(4, int(duration / interval)))
    if total_frames <= 0:
        # some containers (often webm) report no frame count; sample by time instead
        num_samples = max_frames
        step = max(1, int(round(interval * fps)))
        target_indices = set(i * step for i in range(num_samples))
    elif total_frames > num_samples and num_samples > 0:
        step = max(1, total_frames // num_samples)
        target_indices = set(min(i * step, total_frames - 1) for i in range(num_samples))
    else:
        target_indices = set(range(max(1, total_frames)))

    # temporal artifacts (flicker, landmark jitter) only show up between neighbouring frames,
    # so each keyframe also gets a short burst of the frames right after it (~30fps spacing)
    burst_stride = max(1, int(round(fps / 30.0)))
    burst_span = (BURST_LENGTH - 1) * burst_stride
    active_bursts = {}  # keyframe index -> keyframe dict still collecting burst frames
        
    frames = []
    current_idx = 0
    
    while cap.isOpened() and (len(frames) < num_samples or active_bursts):
        # cap.grab() is extremely fast (skips decoding non-target frames)
        grabbed = cap.grab()
        if not grabbed:
            break

        is_target = current_idx in target_indices and len(frames) < num_samples
        burst_owners = [a for a in active_bursts if (current_idx - a) % burst_stride == 0]
            
        if is_target or burst_owners:
            ret, frame = cap.retrieve()
            if ret and frame is not None:
                small = _resize_to_width(frame, BURST_WIDTH)
                for a in burst_owners:
                    active_bursts[a]["burst"].append(small)

                if is_target:
                    h, w = frame.shape[:2]
                    if h > target_height:
                        aspect = w / h
                        new_w = int(target_height * aspect)
                        if new_w % 2 != 0:
                            new_w += 1
                        frame = cv2.resize(frame, (new_w, target_height), interpolation=cv2.INTER_AREA)

                    timestamp = current_idx / fps
                    noise_var = compute_noise_residual(frame)
                    keyframe = {
                        "frame_index": current_idx,
                        "timestamp": round(timestamp, 3),
                        "image": frame,
                        "noise_variance": round(noise_var, 8),
                        "burst": [small]
                    }
                    frames.append(keyframe)
                    active_bursts[current_idx] = keyframe

        # stop collecting for keyframes whose burst is complete
        for a in [a for a in active_bursts if current_idx - a >= burst_span]:
            del active_bursts[a]
        current_idx += 1
        
    cap.release()
    return frames

# extract audio using ffmpeg. returns file path or None if it fails.
def extract_audio(file_path, output_path):
    # check if ffmpeg is installed
    if not shutil.which("ffmpeg"):
        logger.warning("ffmpeg command not found. skipping audio extraction.")
        return None
        
    # -y = overwrite, -vn = no video, acodec mp3 = encode to mp3
    cmd = [
        "ffmpeg", "-y", "-i", file_path,
        "-vn",
        "-acodec", "libmp3lame",
        output_path
    ]
    
    # if wav format is requested
    if output_path.endswith(".wav"):
        cmd = ["ffmpeg", "-y", "-i", file_path, "-vn", output_path]
        
    try:
        # hide log spam unless we need it
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=60)
        if result.returncode != 0:
            logger.error(f"ffmpeg extraction failed (maybe no audio track present): {result.stderr}")
            return None
        return output_path
    except Exception as e:
        logger.error(f"failed to execute ffmpeg subprocess: {e}", exc_info=True)
        return None
