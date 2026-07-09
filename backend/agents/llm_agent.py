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
    scores_list = []
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
            "You are a forensic video expert. Analyze this frame from a video for AI generation (e.g., from tools like Sora, Runway, Pika, Synthesia) or deepfake manipulation.\n"
            "Look closely for: structural distortions, physics violations, temporal morphing/blending artifacts, lighting/shadow mismatches, flat/plasticky textures, or face/eye inconsistencies.\n"
            "CRITICAL requirement: You MUST start your response with a rating in this exact format: [SCORE: X.XX] (e.g., [SCORE: 0.90]) where X.XX is your confidence that this frame is synthetic/AI-generated or manipulated (0.00 is authentic camera footage, 1.00 is fully AI-generated/fake). Then, provide your maximum 2-sentence explanation."
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

        # Use regex to find [SCORE: X.XX] or SCORE: X.XX (case insensitive)
        import re
        match = re.search(r'\[?SCORE:\s*([0-9.]+)(?:/1\.0)?\]?', response_text, re.IGNORECASE)
        if match:
            try:
                score = float(match.group(1))
                # Remove the score prefix from the explanation to clean it up
                explanation = re.sub(r'\[?SCORE:\s*[0-9.]+(?:/1\.0)?\]?', '', response_text, flags=re.IGNORECASE).strip()
            except Exception:
                pass
        else:
            # Fallback: Look for any standalone decimal number between 0.0 and 1.0 in the response text
            numbers = re.findall(r'\b0\.[0-9]+\b|\b1\.0\b', response_text)
            if numbers:
                try:
                    score = float(numbers[0])
                except Exception:
                    pass
            else:
                # Heuristic: Check text keywords to make a guess if the score is missing
                text_lower = response_text.lower()
                if any(word in text_lower for word in ["manipulated", "fake", "ai-generated", "synthetic", "deepfake", "distortions", "artifacts"]):
                    score = 0.85
                elif "authentic" in text_lower or "real" in text_lower:
                    score = 0.05

        ts_str = str(ts)
        frame_explanations[ts_str] = explanation
        scores_list.append(score)
        num_analyzed += 1

    # Average the top 3 highest scores to avoid hiding manipulation in long authentic sections
    if num_analyzed > 0:
        sorted_scores = sorted(scores_list, reverse=True)
        top_scores = sorted_scores[:3]
        llm_score = sum(top_scores) / len(top_scores)
    else:
        llm_score = 0.0

    llm_reasoning = f"Groq analyzed {num_analyzed} frames. Top 3 frame average confidence: {round(llm_score, 2)}"

    return llm_reasoning, frame_explanations, round(llm_score, 4)
