import os
import subprocess
import shutil
import re
import json
import numpy as np
import cv2

# check filename format and video encoder tag using ffprobe.
# e.g., Android camera default names, or if metadata was wiped by generator tools
def check_provenance(file_path):
    filename = os.path.basename(file_path).lower()
    
    # check if filename looks like a camera recording or a social download
    is_camera = bool(re.search(r'(vid|img|pxl|dji|dsc|gopr|mov)_\d+', filename)) or \
                bool(re.search(r'^\d{8}_\d{6}', filename))
                
    is_social = any(platform in filename for platform in ["whatsapp", "snapchat", "tiktok", "facebook", "instagram", "telegram"])
    
    encoder = "unknown"
    metadata_stripped = True
    
    # run ffprobe to extract tags if available
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
            # print error but don't let it crash the run
            print(f"debug: error running ffprobe for provenance check: {e}")
            
    # basic score. 0.5 is neutral/not sure
    provenance_score = 0.5
    if is_camera:
        provenance_score = 0.8
        if encoder != "unknown" and not metadata_stripped:
            provenance_score = 0.95
    elif is_social:
        # social videos are real but compressed, give a decent score
        provenance_score = 0.7
    elif metadata_stripped or encoder == "unknown":
        # warning indicator if metadata is totally wiped
        provenance_score = 0.3
        
    return {
        "provenance_score": provenance_score,
        "is_camera_filename": is_camera,
        "is_social_filename": is_social,
        "encoder": encoder,
        "metadata_stripped": metadata_stripped
    }

# calculate noise variance. real camera sensors have thermal/lens noise.
# synthetic images (AI generators) tend to be way too smooth.
def compute_noise_residual(frame):
    # standard grayscale conversion
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # blur the image slightly to make a smooth version
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    
    # get difference between original and blurred (the residual noise)
    residual = gray.astype(np.float32) - blurred.astype(np.float32)
    
    # calculate noise variance
    return float(np.var(residual))

# helper to pull out all the useful metadata from opencv
# we need this for validation and also to return to the frontend later
def get_video_metadata(file_path):
    cap = cv2.VideoCapture(file_path)
    if not cap.isOpened():
        raise ValueError("could not open video file to read metadata")
    
    # query the properties we need
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # convert fourcc int into a readable string like 'mp4v' or 'h264'
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    codec = ""
    for i in range(4):
        codec += chr((fourcc >> (8 * i)) & 0xFF)
        
    duration = 0.0
    if fps > 0:
        duration = total_frames / fps
        
    cap.release()
    
    # let's try calculating the bitrate since opencv doesn't give it directly
    bitrate = 0
    try:
        file_size = os.path.getsize(file_path)
        if duration > 0:
            # size in bits divided by duration
            bitrate = int((file_size * 8) / duration)
    except Exception:
        # if file size check fails, just keep bitrate as 0
        pass
        
    # run provenance check
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

# checks if the video file exists, is a format we support, and is short enough
# we don't want people uploading 2-hour movies and killing our free tier server
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

# resizes the video to 480p while keeping the aspect ratio correct
# writes it to a new file so we can save bandwidth/processing power
def downscale_video(file_path, output_path, target_height=480):
    cap = cv2.VideoCapture(file_path)
    if not cap.isOpened():
        raise ValueError("could not open video to downscale")
        
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # if it's already smaller than target_height, don't upscale it
    if height <= target_height:
        # write using original dimensions
        new_height = height
        new_width = width
    else:
        # compute width using aspect ratio
        aspect = width / height
        new_height = target_height
        new_width = int(target_height * aspect)
        
        # make sure new_width is even because some encoders crash with odd widths
        if new_width % 2 != 0:
            new_width += 1
            
    # we'll use 'mp4v' fourcc as it's standard and works almost everywhere
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (new_width, new_height))
    
    # read frame by frame, resize, and write
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        resized = cv2.resize(frame, (new_width, new_height))
        out.write(resized)
        
    cap.release()
    out.release()
    return output_path

# extracts frames from video at a constant interval (e.g. 0.5 seconds)
# returns list of dicts with frame index, timestamp, and numpy image
# does NOT write them to disk to keep processing fast and private
def extract_frames(file_path, interval=0.5):
    cap = cv2.VideoCapture(file_path)
    if not cap.isOpened():
        raise ValueError("could not open video to extract frames")
        
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        # fallback in case fps reading is broken
        fps = 30.0
        
    # calculate how many frames to skip between extractions
    frame_skip = max(1, int(fps * interval))
    
    frames = []
    frame_idx = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # extract every Nth frame
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

# extracts audio track from video using ffmpeg command line tool
# returns path to output file or None if it fails
def extract_audio(file_path, output_path):
    # first check if ffmpeg is even on the system
    if not shutil.which("ffmpeg"):
        print("warning: ffmpeg command not found. skipping audio extraction.")
        return None
        
    # ffmpeg command options:
    # -y (overwrite output file if it exists)
    # -vn (disable video recording)
    # -acodec libmp3lame (encode as mp3 for easy handling)
    cmd = [
        "ffmpeg", "-y", "-i", file_path,
        "-vn",
        "-acodec", "libmp3lame",
        output_path
    ]
    
    # if user asked for wav, let ffmpeg handle format automatically
    if output_path.endswith(".wav"):
        cmd = ["ffmpeg", "-y", "-i", file_path, "-vn", output_path]
        
    try:
        # run command and hide stdout/stderr logs unless we need to debug
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode != 0:
            print(f"ffmpeg extraction failed (maybe no audio track present): {result.stderr}")
            return None
        return output_path
    except Exception as e:
        print(f"failed to execute ffmpeg subprocess: {e}")
        return None

# testing script to make sure functions work locally
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python preprocessing.py <path_to_video_file>")
        sys.exit(1)
        
    test_video = sys.argv[1]
    print(f"Testing preprocessing on: {test_video}")
    
    try:
        # 1. validate
        print("\n--- Validating ---")
        meta = validate_video(test_video)
        print("Metadata:", meta)
        
        # 2. downscale
        print("\n--- Downscaling ---")
        out_downscaled = "temp_downscaled.mp4"
        downscale_video(test_video, out_downscaled)
        print(f"Downscaled video saved to {out_downscaled}")
        
        # check downscaled metadata
        downscaled_meta = get_video_metadata(out_downscaled)
        print("Downscaled Metadata:", downscaled_meta)
        
        # 3. extract frames
        print("\n--- Extracting Frames ---")
        frames = extract_frames(out_downscaled, interval=0.5)
        print(f"Extracted {len(frames)} frames in total.")
        if len(frames) > 0:
            first_frame = frames[0]
            print(f"First frame details - Index: {first_frame['frame_index']}, Timestamp: {first_frame['timestamp']}s, Shape: {first_frame['image'].shape}, Noise Variance: {first_frame['noise_variance']}")
            
        # 4. extract audio
        print("\n--- Extracting Audio ---")
        out_audio = "temp_audio.mp3"
        audio_path = extract_audio(test_video, out_audio)
        if audio_path:
            print(f"Audio extracted successfully to {audio_path}")
        else:
            print("Audio extraction was skipped or failed.")
            
        # clean up temp files
        if os.path.exists(out_downscaled):
            os.remove(out_downscaled)
            print(f"Cleaned up {out_downscaled}")
        if audio_path and os.path.exists(audio_path):
            os.remove(audio_path)
            print(f"Cleaned up {audio_path}")
            
        print("\nAll preprocessing tests completed successfully!")
        
    except Exception as err:
        print(f"Error occurred during test: {err}")
        sys.exit(1)
