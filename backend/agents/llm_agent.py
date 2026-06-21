from PIL import Image
import os

# import google genai if available, but wrap in try/except so it doesn't crash if missing
try:
    from google import genai
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False


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


def analyze_with_gemini(suspicious_frames):
    # run gemini 2.5 flash on suspicious frames to find deepfake cues
    # if no frames to analyze, return empty results
    if not suspicious_frames:
        return (
            "no suspicious frames flagged for analysis",
            {},
            0.0
        )

    # check if API key is in environment
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GENAI_API_KEY")
    if not GENAI_AVAILABLE or not api_key:
        print("warning: gemini api key not found or library missing. using offline fallback.")
        # create fallback explanations so pipeline continues
        frame_explanations = {}
        for f in suspicious_frames:
            ts_str = str(round(f["timestamp"], 3))
            frame_explanations[ts_str] = "AI explanation unavailable (offline fallback mode)."

        reasoning = "gemini analysis was skipped because no api key was found"
        return reasoning, frame_explanations, 0.0

    try:
        # initialize the google-genai client
        client = genai.Client(api_key=api_key)

        frame_explanations = {}
        total_fake_conf = 0.0
        num_analyzed = 0

        print(f"sending {len(suspicious_frames)} frames to gemini 2.5 flash...")

        for f in suspicious_frames:
            ts = round(f["timestamp"], 3)
            img_array = f["image"]

            # convert BGR to RGB and PIL
            rgb = img_array[:, :, ::-1]
            pil_img = Image.fromarray(rgb)

            # craft a structured prompt for deepfake cues
            prompt = (
                "You are a forensic video expert. Analyze this frame from a video for deepfake manipulation. "
                "Look for: lighting inconsistencies, unnatural blending around boundaries (face, hair, eyes), "
                "shadow direction mismatches, or skin texture anomalies. "
                "Reply in 2 sentences max. Start your response with a rating in this format: [SCORE: X.XX] "
                "where X.XX is your confidence that this image is fake (0.00 is authentic, 1.00 is manipulated)."
            )

            # call the gemini api
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=[pil_img, prompt]
            )

            response_text = response.text or ""
            print(f"gemini response for t={ts}s: {response_text.strip()}")

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
                    print(f"could not parse score from gemini output: {parse_err}")

            ts_str = str(ts)
            frame_explanations[ts_str] = explanation
            total_fake_conf += score
            num_analyzed += 1

        # average score of gemini ratings
        llm_score = total_fake_conf / num_analyzed if num_analyzed > 0 else 0.0
        llm_reasoning = f"gemini analyzed {num_analyzed} frames. average confidence: {round(llm_score, 2)}"

        return llm_reasoning, frame_explanations, round(llm_score, 4)

    except Exception as e:
        print(f"error calling gemini api: {e}. fallback to offline data.")
        frame_explanations = {}
        for f in suspicious_frames:
            ts_str = str(round(f["timestamp"], 3))
            frame_explanations[ts_str] = "AI explanation unavailable (call failed)."

        reasoning = f"gemini failed with error: {e}"
        return reasoning, frame_explanations, 0.0
