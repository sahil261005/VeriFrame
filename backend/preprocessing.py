import os
import subprocess
import shutil
import re
import json
import numpy as np
import cv2

def check_provenance(file_path):
    filename = os.path.basename(file_path).lower()
    
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
    c2pa_compliant = False
    
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
                    
                    for k, v in format_tags.items():
                        k_lower = str(k).lower()
                        v_lower = str(v).lower()
                        if "c2pa" in k_lower or "jumb" in k_lower or "provenance" in k_lower or "c2pa" in v_lower:
                            c2pa_compliant = True
                            break
        except Exception as e:
            print(f"debug: error running ffprobe for provenance check: {e}")
            
    provenance_score = 0.5
    if c2pa_compliant:
        provenance_score = 0.98
    elif is_camera:
        provenance_score = 0.8
        if encoder != "unknown" and not metadata_stripped:
            provenance_score = 0.95
    elif is_social:
        provenance_score = 0.7
    elif metadata_stripped or encoder == "unknown":
        provenance_score = 0.3
        
    return {
        "provenance_score": provenance_score,
        "is_camera_filename": is_camera,
        "is_social_filename": is_social,
        "encoder": encoder,
        "metadata_stripped": metadata_stripped,
        "c2pa_compliant": c2pa_compliant
    }

def get_video_metadata(file_path):
    cap = cv2.VideoCapture(file_path)
    if not cap.isOpened():
        raise ValueError("could not open video file to read metadata")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    codec = ""
    for i in range(4):
        codec += chr((fourcc >> (8 * i)) & 0xFF)
        
    duration = 0.0
    if fps > 0:
        duration = total_frames / fps
        
    cap.release()
    
    bitrate = 0
    try:
        file_size = os.path.getsize(file_path)
        if duration > 0:
            bitrate = int((file_size * 8) / duration)
    except Exception:
        pass
        
    provenance = check_provenance(file_path)
    
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

def validate_video(file_path):
    if not os.path.exists(file_path):
        raise ValueError("video file does not exist")
        
    ext = os.path.splitext(file_path)[1].lower().replace(".", "")
    allowed = ["mp4", "avi", "mov", "webm"]
    if ext not in allowed:
        raise ValueError(f"unsupported format: {ext}. we only support mp4, avi, mov, webm")
        
    meta = get_video_metadata(file_path)
    if meta["duration"] > 30.0:
        raise ValueError(f"video duration is {meta['duration']}s, which exceeds the 30-second limit")
        
    return meta

def downscale_video(file_path, output_path, target_height=480):
    cap = cv2.VideoCapture(file_path)
    if not cap.isOpened():
        raise ValueError("could not open video to downscale")
        
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    if height <= target_height:
        new_height = height
        new_width = width
    else:
        aspect = width / height
        new_height = target_height
        new_width = int(target_height * aspect)
        
        if new_width % 2 != 0:
            new_width += 1
            
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (new_width, new_height))
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        resized = cv2.resize(frame, (new_width, new_height))
        out.write(resized)
        
    cap.release()
    out.release()
    return output_path

def extract_frames(file_path, interval=0.5):
    cap = cv2.VideoCapture(file_path)
    if not cap.isOpened():
        raise ValueError("could not open video to extract frames")
        
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0
        
    frame_skip = max(1, int(fps * interval))
    
    frames = []
    frame_idx = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
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

def extract_audio(file_path, output_path):
    if not shutil.which("ffmpeg"):
        print("warning: ffmpeg command not found. skipping audio extraction.")
        return None
        
    cmd = [
        "ffmpeg", "-y", "-i", file_path,
        "-vn",
        "-acodec", "libmp3lame",
        output_path
    ]
    
    if output_path.endswith(".wav"):
        cmd = ["ffmpeg", "-y", "-i", file_path, "-vn", output_path]
        
    try:
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode != 0:
            print(f"ffmpeg extraction failed (maybe no audio track present): {result.stderr}")
            return None
        return output_path
    except Exception as e:
        print(f"failed to execute ffmpeg subprocess: {e}")
        return None
