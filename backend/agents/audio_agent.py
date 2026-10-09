import os
import subprocess
import wave
import numpy as np
import logging

logger = logging.getLogger(__name__)


def extract_audio_wav(video_path, output_wav):
    """
    uses ffmpeg to pull the audio out as a 16kHz mono wav.
    returns True if it worked, False if the video has no audio
    """
    try:
        cmd = [
            "ffmpeg", "-y", "-i", video_path,
            "-vn", "-ac", "1", "-ar", "16000",
            output_wav
        ]
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=60)
        if res.returncode == 0 and os.path.exists(output_wav) and os.path.getsize(output_wav) > 1000:
            return True
    except Exception as e:
        logger.warning(f"Audio extraction error: {e}")
    return False


def analyze_spectral_cutoffs(audio_samples, sample_rate=16000):
    """
    AI voices like elevenlabs or bark often have a sharp cutoff in the high frequencies.
    real recordings still have some hiss and breath above 6.5kHz
    """
    if len(audio_samples) < sample_rate:
        return 0.2, "Audio too short for spectral analysis."

    # fft to get the energy at each frequency
    fft_vals = np.abs(np.fft.rfft(audio_samples))
    freqs = np.fft.rfftfreq(len(audio_samples), 1.0 / sample_rate)

    total_energy = float(np.sum(fft_vals)) + 1e-6
    # how much energy is above 6500 Hz
    high_freq_mask = freqs > 6500
    high_freq_energy = float(np.sum(fft_vals[high_freq_mask]))
    high_freq_ratio = high_freq_energy / total_energy

    # AI tts usually cuts off hard and has almost no high freq energy, under 0.015
    if high_freq_ratio < 0.015:
        score = 0.80
        desc = "Unnatural high-frequency spectral cutoff detected (< 6.5kHz). Characteristic of synthetic TTS vocoders."
    elif high_freq_ratio < 0.035:
        score = 0.45
        desc = "Low high-frequency energy detected. Possible compression or voice synthesis."
    else:
        score = 0.10
        desc = "Natural acoustic frequency distribution spanning full spectrum."

    return score, desc


def analyze_voice_cadence(audio_samples, sample_rate=16000):
    """
    checks if the voice volume is too flat and robotic, like no breathing pauses
    """
    chunk_size = int(sample_rate * 0.1)  # 100ms chunks
    if len(audio_samples) < chunk_size * 5:
        return 0.15, "Insufficient speech duration."

    num_chunks = len(audio_samples) // chunk_size
    rms_values = []
    for i in range(num_chunks):
        chunk = audio_samples[i * chunk_size:(i + 1) * chunk_size]
        rms = np.sqrt(np.mean(chunk.astype(np.float32) ** 2))
        rms_values.append(rms)

    rms_arr = np.array(rms_values)
    mean_rms = float(np.mean(rms_arr)) + 1e-6
    var_rms = float(np.var(rms_arr / mean_rms))

    # real speech has pauses between syllables so var_rms is usually above 0.35
    # a fake flat voice has really low variance
    if var_rms < 0.10 and mean_rms > 100:
        score = 0.75
        desc = "Unnaturally flat speech volume envelope without natural breathing micro-pauses."
    else:
        score = 0.15
        desc = "Natural dynamic speech rhythm and syllable variance."

    return score, desc


def check_lip_sync(audio_samples, sample_rate, frames):
    """
    compares loud parts of the audio with the keyframes.
    if the frame looks static while the audio is loud we count it as desync
    """
    if not frames or len(audio_samples) < sample_rate:
        return 0.2, "Not enough frames for lip-sync correlation."

    desync_count = 0
    total_checked = 0

    for f in frames:
        ts = f.get("timestamp", 0.0)
        # take 200ms of audio around the frame
        center_sample = int(ts * sample_rate)
        start = max(0, center_sample - int(sample_rate * 0.1))
        end = min(len(audio_samples), center_sample + int(sample_rate * 0.1))

        if end > start:
            window = audio_samples[start:end]
            rms = np.sqrt(np.mean(window.astype(np.float32) ** 2))
            is_audio_active = rms > 300  # loud enough to count as speech

            total_checked += 1
            # simple check, flag it if audio is loud but the frame is static
            if is_audio_active and f.get("noise_variance", 0) < 0.00002:
                desync_count += 1

    if total_checked > 0 and (desync_count / total_checked) > 0.5:
        return 0.70, f"Mouth-to-voice desynchronization detected across {desync_count}/{total_checked} keyframes."
    
    return 0.15, "Voice energy and visual keyframes appear temporally synchronized."


def run_audio_analysis(video_path, frames):
    """
    main function for the audio agent, returns (score, details, has_audio)
    """
    if not video_path or not os.path.exists(video_path):
        return 0.0, {"summary": "No video file available for audio extraction.", "has_audio": False}, False

    wav_path = f"{video_path}_temp_audio.wav"
    has_audio = extract_audio_wav(video_path, wav_path)

    if not has_audio:
        if os.path.exists(wav_path):
            try:
                os.remove(wav_path)
            except Exception:
                pass
        return 0.0, {"summary": "Video has no audio track (silent clip).", "has_audio": False}, False

    try:
        with wave.open(wav_path, 'r') as wf:
            n_frames = wf.getnframes()
            sample_rate = wf.getframerate()
            raw_bytes = wf.readframes(n_frames)
            audio_samples = np.frombuffer(raw_bytes, dtype=np.int16)

        # run the 3 checks
        spectral_score, spectral_desc = analyze_spectral_cutoffs(audio_samples, sample_rate)
        cadence_score, cadence_desc = analyze_voice_cadence(audio_samples, sample_rate)
        lipsync_score, lipsync_desc = check_lip_sync(audio_samples, sample_rate, frames)

        final_audio_score = round(float(0.45 * spectral_score + 0.30 * cadence_score + 0.25 * lipsync_score), 4)

        details = {
            "has_audio": True,
            "spectral_score": spectral_score,
            "spectral_summary": spectral_desc,
            "cadence_score": cadence_score,
            "cadence_summary": cadence_desc,
            "lipsync_score": lipsync_score,
            "lipsync_summary": lipsync_desc,
            "summary": f"Spectral: {spectral_desc} Cadence: {cadence_desc}"
        }

        return final_audio_score, details, True

    except Exception as e:
        logger.error(f"Error analyzing audio: {e}", exc_info=True)
        return 0.0, {"summary": f"Audio processing error: {e}", "has_audio": False}, False
    finally:
        # delete the temp wav
        if os.path.exists(wav_path):
            try:
                os.remove(wav_path)
            except Exception:
                pass
