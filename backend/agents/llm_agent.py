import os
import io
import base64
from PIL import Image
from groq import Groq


def pick_suspicious_frames(visual_flagged, temporal_flagged, all_frames):
    flagged_times = set()
    
    for f in visual_flagged:
        flagged_times.add(round(f["timestamp"], 3))
        
    for t in temporal_flagged:
        flagged_times.add(round(t, 3))
        
    sorted_times = sorted(list(flagged_times))
    suspicious_frames = []
    
    for t in sorted_times:
        for frame in all_frames:
            if abs(frame["timestamp"] - t) < 0.01:
                suspicious_frames.append(frame)
                break
                
    return suspicious_frames[:8]


def analyze_with_llm(suspicious_frames):
    if not suspicious_frames:
        return "No suspicious frames flagged for analysis", {}, 0.0

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        frame_explanations = {}
        for f in suspicious_frames:
            ts_str = str(round(f["timestamp"], 3))
            frame_explanations[ts_str] = "Analysis skipped (no GROQ_API_KEY set)."
        return "Groq analysis skipped because GROQ_API_KEY is missing", frame_explanations, 0.0

    client = Groq(api_key=api_key)
    frame_explanations = {}
    total_fake_conf = 0.0
    num_analyzed = 0

    for f in suspicious_frames:
        ts = round(f["timestamp"], 3)
        img_array = f["image"]

        # Convert BGR to RGB and PIL Image
        rgb = img_array[:, :, ::-1]
        pil_img = Image.fromarray(rgb)

        # Convert PIL Image to base64 jpeg
        buffered = io.BytesIO()
        pil_img.save(buffered, format="JPEG")
        base64_image = base64.b64encode(buffered.getvalue()).decode('utf-8')

        prompt = (
            "You are a forensic video expert. Analyze this frame from a video for deepfake manipulation. "
            "Look for: lighting inconsistencies, unnatural blending around boundaries (face, hair, eyes), "
            "shadow direction mismatches, or skin texture anomalies. "
            "Reply in 2 sentences max. Start your response with a rating in this format: [SCORE: X.XX] "
            "where X.XX is your confidence that this image is fake (0.00 is authentic, 1.00 is manipulated)."
        )

        response = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}",
                            },
                        },
                    ],
                }
            ],
        )

        response_text = response.choices[0].message.content or ""
        score = 0.1
        explanation = response_text

        if "[SCORE:" in response_text:
            parts = response_text.split("[SCORE:")
            score_part = parts[1].split("]")[0].strip()
            score = float(score_part)
            explanation = parts[0] + parts[1].split("]")[1]
            explanation = explanation.strip()

        ts_str = str(ts)
        frame_explanations[ts_str] = explanation
        total_fake_conf += score
        num_analyzed += 1

    llm_score = total_fake_conf / num_analyzed if num_analyzed > 0 else 0.0
    llm_reasoning = f"Groq analyzed {num_analyzed} frames. Average confidence: {round(llm_score, 2)}"

    return llm_reasoning, frame_explanations, round(llm_score, 4)
