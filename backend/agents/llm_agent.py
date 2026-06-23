from PIL import Image
import os
import io
import base64

# import groq if available, but wrap in try/except so it doesn't crash if missing
try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False


def pick_suspicious_frames(visual_flagged, temporal_flagged, all_frames):
    # merge visual and temporal flagged frames, remove duplicates, cap at 8 max
    flagged_times = set()

    # add timestamps from visual agent
    for f in visual_flagged:
        flagged_times.add(round(f["timestamp"], 3))

    # add timestamps from temporal agent
    for t in temporal_flagged:
        flagged_times.add(round(t, 3))

    # convert to sorted list
    sorted_times = sorted(list(flagged_times))

    suspicious_frames = []
    # find the frame dicts in all_frames that match our suspicious timestamps
    for t in sorted_times:
        for frame in all_frames:
            if abs(frame["timestamp"] - t) < 0.01:
                suspicious_frames.append(frame)
                break

    # cap at 8 frames to avoid rate limits / huge api payloads
    return suspicious_frames[:8]


def analyze_with_llm(suspicious_frames):
    # run llama-3.2-11b-vision-preview on suspicious frames to find deepfake cues
    # if no frames to analyze, return empty results
    if not suspicious_frames:
        return (
            "no suspicious frames flagged for analysis",
            {},
            0.0
        )

    # check if API key is in environment
    api_key = os.environ.get("GROQ_API_KEY")
    if not GROQ_AVAILABLE or not api_key:
        print("warning: groq api key not found or library missing. using offline fallback.")
        # create fallback explanations so pipeline continues
        frame_explanations = {}
        for f in suspicious_frames:
            ts_str = str(round(f["timestamp"], 3))
            frame_explanations[ts_str] = "AI explanation unavailable (offline fallback mode)."

        reasoning = "groq analysis was skipped because no api key was found"
        return reasoning, frame_explanations, 0.0

    try:
        # initialize the groq client
        client = Groq(api_key=api_key)

        frame_explanations = {}
        total_fake_conf = 0.0
        num_analyzed = 0

        print(f"sending {len(suspicious_frames)} frames to llama-3.2-11b-vision-preview via Groq...")

        for f in suspicious_frames:
            ts = round(f["timestamp"], 3)
            img_array = f["image"]

            # convert BGR to RGB and PIL
            rgb = img_array[:, :, ::-1]
            pil_img = Image.fromarray(rgb)

            # convert PIL Image to base64 jpeg
            buffered = io.BytesIO()
            pil_img.save(buffered, format="JPEG")
            base64_image = base64.b64encode(buffered.getvalue()).decode('utf-8')

            # craft a structured prompt for deepfake cues
            prompt = (
                "You are a forensic video expert. Analyze this frame from a video for deepfake manipulation. "
                "Look for: lighting inconsistencies, unnatural blending around boundaries (face, hair, eyes), "
                "shadow direction mismatches, or skin texture anomalies. "
                "Reply in 2 sentences max. Start your response with a rating in this format: [SCORE: X.XX] "
                "where X.XX is your confidence that this image is fake (0.00 is authentic, 1.00 is manipulated)."
            )

            # call the groq api using llama-3.2-11b-vision-preview
            response = client.chat.completions.create(
                model="llama-3.2-11b-vision-preview",
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
            print(f"groq response for t={ts}s: {response_text.strip()}")

            # parse score from the format: [SCORE: X.XX]
            # default score is 0.1 if parsing fails
            score = 0.1
            explanation = response_text

            if "[SCORE:" in response_text:
                try:
                    parts = response_text.split("[SCORE:")
                    score_part = parts[1].split("]")[0].strip()
                    score = float(score_part)
                    # remove the score tag from the final text explanation
                    explanation = parts[0] + parts[1].split("]")[1]
                    explanation = explanation.strip()
                except Exception as parse_err:
                    print(f"could not parse score from groq output: {parse_err}")

            ts_str = str(ts)
            frame_explanations[ts_str] = explanation
            total_fake_conf += score
            num_analyzed += 1

        # average score of groq ratings
        llm_score = total_fake_conf / num_analyzed if num_analyzed > 0 else 0.0
        llm_reasoning = f"groq analyzed {num_analyzed} frames. average confidence: {round(llm_score, 2)}"

        return llm_reasoning, frame_explanations, round(llm_score, 4)

    except Exception as e:
        print(f"error calling groq api: {e}. fallback to offline data.")
        frame_explanations = {}
        for f in suspicious_frames:
            ts_str = str(round(f["timestamp"], 3))
            frame_explanations[ts_str] = "AI explanation unavailable (call failed)."

        reasoning = f"groq failed with error: {e}"
        return reasoning, frame_explanations, 0.0
