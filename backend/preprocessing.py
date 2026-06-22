import os
import subprocess
import shutil
import re
import json
import numpy as np
import cv2

# check video filename and encoder tags to see if they look legit or got wiped
def check_provenance(file_path):
    filename = os.path.basename(file_path).lower()
    
    # does it look like a phone camera name? or a social media app name?
    is_camera = False
    if re.search(r'(vid|img|pxl|dji|dsc|gopr|mov)_\d+', filename) or re.search(r'^\d{8}_\d{6}', filename):
        is_camera = True
                
    is_social = False
    for platform in ["whatsapp", "snapchat", "tiktok", "facebook", "instagram", "telegram"]:
        if platform in filename:
            is_social = True
            break
    
    encoder = "unknown"
    metadata_stripped = True
    
    # use ffprobe to get the metadata tags if its installed
    if shutil.which("ffprobe"):
        try:
            cmd = [
                "ffprobe", "-v", "quiet", "-print_format", "json",
                "-show_format", "-show_streams", file_path
            ]
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if result.returncode == 0:
                data = json.loads(result.stdout)
                format_tags = data.get("format", {}).get("tags", {})
                if format_tags:
                    encoder = format_tags.get("encoder", "unknown").lower()
                    metadata_stripped = False
        except Exception as e:
            # print but dont crash the whole run if ffprobe fails
            print(f"debug: error running ffprobe for provenance check: {e}")
            
    # just a simple score between 0 and 1. 0.5 is neutral
    provenance_score = 0.5
    if is_camera:
        provenance_score = 0.8
        if encoder != "unknown" and not metadata_stripped:
            provenance_score = 0.95
    elif is_social:
        # social media compress videos a lot but they are usually real
        provenance_score = 0.7
    elif metadata_stripped or encoder == "unknown":
        # bad sign if metadata is totally empty
        provenance_score = 0.3
        
    return {
        "provenance_score": provenance_score,
        "is_camera_filename": is_camera,
        "is_social_filename": is_social,
        "encoder": encoder,
        "metadata_stripped": metadata_stripped
    }

# real cameras have sensor noise but AI generated frames are usually super clean and smooth
def compute_noise_residual(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    residual = gray.astype(np.float32) - blurred.astype(np.float32)
    return float(np.var(residual))

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
        
    return {
        "fps": round(fps, 2),
        "duration": round(duration, 2),
        "width": width,
        "height": height,
        "codec": codec,
        "total_frames": total_frames,
        "bitrate": bitrate,
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
        
    return meta

# downscales to 480p so we save bandwidth and processing power.
# keeps aspect ratio same
def downscale_video(file_path, output_path, target_height=480):
    cap = cv2.VideoCapture(file_path)
    if not cap.isOpened():
        raise ValueError("could not open video to downscale")
        
    fps = cap.get(cv2.CAP_PROP_FPS)
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
        resized = cv2.resize(frame, (new_width, new_height))
        out.write(resized)
        
    cap.release()
    out.release()
    return output_path

# extract frames every X seconds. does not write to disk so its fast.
# returns dicts with timestamp, index, and numpy array
def extract_frames(file_path, interval=0.5):
    cap = cv2.VideoCapture(file_path)
    if not cap.isOpened():
        raise ValueError("could not open video to extract frames")
        
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        # fallback if fps is broken
        fps = 30.0
        
    # how many frames to skip
    frame_skip = max(1, int(fps * interval))
    
    frames = []
    frame_idx = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # pull the frame if its on the interval skip
        if frame_idx % frame_skip == 0:
            timestamp = frame_idx / fps
            noise_var = compute_noise_residual(frame)
            frames.append({
                "frame_index": frame_idx,
                "timestamp": round(timestamp, 3),
                "image": frame,
                "noise_variance": round(noise_var, 4)
            })
        frame_idx += 1
        
    cap.release()
    return frames

# extract audio using ffmpeg. returns file path or None if it fails.
def extract_audio(file_path, output_path):
    # check if ffmpeg is installed
    if not shutil.which("ffmpeg"):
        print("warning: ffmpeg command not found. skipping audio extraction.")
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
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode != 0:
            print(f"ffmpeg extraction failed (maybe no audio track present): {result.stderr}")
            return None
        return output_path
    except Exception as e:
        print(f"failed to execute ffmpeg subprocess: {e}")
        return None
