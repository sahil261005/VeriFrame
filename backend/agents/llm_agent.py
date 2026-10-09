import os
import threading
import io
import base64
import json
import re
import logging
from PIL import Image
import cv2
from concurrent.futures import ThreadPoolExecutor

# import this at startup, not on the first request, because it takes over a second on a small cpu server
from google import genai
from google.genai import types
from agents.tools import TOOL_REGISTRY

logger = logging.getLogger(__name__)


class LLMUnavailableError(Exception):
    """raised when theres no llm api key set, so the agent gets skipped instead of scored"""


_usage_lock = threading.Lock()

# vision model for the groq fallback. groq removed llama 4 scout (it gave a 404) so now its an env var.
# qwen/qwen3.8-27b takes images and sticks to the [SCORE: X.XX] format
GROQ_VISION_MODEL = os.environ.get("GROQ_VISION_MODEL", "qwen/qwen3.8-27b")

# price in USD per 1M tokens from the providers pricing page, set these to get a cost per video.
# if they arent set we still count tokens but dont calculate the cost
_PRICE_ENV = {
    "gemini": ("GEMINI_INPUT_USD_PER_M", "GEMINI_OUTPUT_USD_PER_M"),
    "groq": ("GROQ_INPUT_USD_PER_M", "GROQ_OUTPUT_USD_PER_M"),
}


def _record_usage(usage, provider, input_tokens, output_tokens):
    """adds the token counts from one api call to the usage dict, locked because groq calls run in threads"""
    if usage is None:
        return
    with _usage_lock:
        usage["llm_calls"] = usage.get("llm_calls", 0) + 1
        usage[f"{provider}_input_tokens"] = usage.get(f"{provider}_input_tokens", 0) + int(input_tokens or 0)
        usage[f"{provider}_output_tokens"] = usage.get(f"{provider}_output_tokens", 0) + int(output_tokens or 0)


def estimate_cost_usd(usage):
    """how much the tokens cost in dollars, None if the prices arent set"""
    total = 0.0
    priced = False
    for provider, (in_env, out_env) in _PRICE_ENV.items():
        tokens_in = usage.get(f"{provider}_input_tokens", 0)
        tokens_out = usage.get(f"{provider}_output_tokens", 0)
        if not tokens_in and not tokens_out:
            continue
        in_price, out_price = os.environ.get(in_env), os.environ.get(out_env)
        if in_price is None or out_price is None:
            return None
        total += tokens_in / 1e6 * float(in_price) + tokens_out / 1e6 * float(out_price)
        priced = True
    return round(total, 6) if priced else 0.0


def _clamp_score(value):
    return min(max(float(value), 0.0), 1.0)


def pick_suspicious_frames(visual_flagged, temporal_flagged, all_frames, max_count=8):
    # visual_flagged is already sorted most suspicious first. we alternate it with the temporal ones
    # so cutting to max_count keeps the best frames from both agents, then fill the rest with
    # evenly spaced frames and sort by time
    by_ts = {round(f["timestamp"], 3): f for f in all_frames}

    def resolve(t):
        t = round(t, 3)
        if t in by_ts:
            return by_ts[t]
        for frame in all_frames:
            if abs(frame["timestamp"] - t) < 0.01:
                return frame
        return None

    visual_ts = [f["timestamp"] for f in visual_flagged]
    temporal_ts = list(temporal_flagged)
    prioritized = []
    for i in range(max(len(visual_ts), len(temporal_ts))):
        if i < len(visual_ts):
            prioritized.append(visual_ts[i])
        if i < len(temporal_ts):
            prioritized.append(temporal_ts[i])

    suspicious_frames = []
    existing_ts = set()
    for t in prioritized:
        frame = resolve(t)
        if frame is None:
            continue
        ts = round(frame["timestamp"], 3)
        if ts in existing_ts:
            continue
        suspicious_frames.append(frame)
        existing_ts.add(ts)
        if len(suspicious_frames) >= max_count:
            break

    if len(suspicious_frames) < max_count and len(all_frames) > 0:
        step = max(1, len(all_frames) // max_count)
        for i in range(0, len(all_frames), step):
            f = all_frames[i]
            ts = round(f["timestamp"], 3)
            if ts not in existing_ts:
                suspicious_frames.append(f)
                existing_ts.add(ts)
            if len(suspicious_frames) >= max_count:
                break

    suspicious_frames = suspicious_frames[:max_count]
    suspicious_frames.sort(key=lambda f: f["timestamp"])
    return suspicious_frames


def run_tools_for_frame(frame_data, all_frames=None, metadata=None):
    # runs our tools on one frame and collects the results
    # reuses the noise variance we already computed, saves 5+ seconds of cpu
    tool_results = {}
    tools_called = []

    # 1. noise tool, instant since it just reads noise_variance from the frame
    try:
        noise_var = frame_data.get("noise_variance")
        if noise_var is not None:
            if noise_var < 0.00003:
                risk = "high (unnaturally smooth noise, characteristic of AI generation)"
            elif noise_var < 0.00008:
                risk = "medium (low sensor noise)"
            else:
                risk = "low (natural camera sensor noise pattern)"

            tool_results["noise_pattern"] = {
                "tool": "analyze_noise_pattern",
                "noise_variance": round(noise_var, 8),
                "noise_risk": risk,
                "summary": f"Normalized noise variance: {noise_var:.7f} ({risk})."
            }
        else:
            res = TOOL_REGISTRY["analyze_noise_pattern"]["function"](frame_data)
            tool_results["noise_pattern"] = res
        tools_called.append("analyze_noise_pattern")
    except Exception as e:
        logger.warning(f"tool analyze_noise_pattern failed: {e}")

    # 2. optical flow tool, only if we have all_frames
    if all_frames and len(all_frames) > 1:
        try:
            res = TOOL_REGISTRY["compare_adjacent_frames"]["function"](frame_data, all_frames)
            tool_results["adjacent_frames"] = res
            tools_called.append("compare_adjacent_frames")
        except Exception as e:
            logger.warning(f"tool compare_adjacent_frames failed: {e}")

    # 3. metadata tool, only if we have metadata
    if metadata and "metadata" not in tool_results:
        try:
            res = TOOL_REGISTRY["check_metadata"]["function"](metadata)
            tool_results["metadata"] = res
            tools_called.append("check_metadata")
        except Exception as e:
            logger.warning(f"tool check_metadata failed: {e}")

    return tool_results, tools_called


def analyze_with_gemini(suspicious_frames, reflection_prompt="", metadata=None, all_frames=None, api_key=None, audio_details=None, usage=None):
    # sends all the frames to gemini 3.5 flash-lite in one request and gets json back

    # need a timeout or a stuck gemini call blocks the only analysis slot forever.
    # its 60s (value is in ms), after that we fall back to groq
    client = genai.Client(api_key=api_key, http_options=types.HttpOptions(timeout=60000))
    contents = []
    tool_summaries = []
    all_tools_used = set()

    # run the tools on all frames at once with threads, they dont depend on each other
    workers = min(len(suspicious_frames), 6)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        tool_results_list = list(pool.map(
            lambda f: run_tools_for_frame(f, all_frames=all_frames, metadata=metadata),
            suspicious_frames
        ))

    for idx, (f, (tool_results, tools_called)) in enumerate(zip(suspicious_frames, tool_results_list)):
        ts = round(f["timestamp"], 3)
        all_tools_used.update(tools_called)

        # 640px wide so small stuff like text, hands and skin texture is still visible, 384px blurred it out
        rgb = cv2.cvtColor(f["image"], cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        target_w = 640
        target_h = max(1, int(h * (target_w / w)))
        small_rgb = cv2.resize(rgb, (target_w, target_h), interpolation=cv2.INTER_AREA)
        pil_img = Image.fromarray(small_rgb)

        obs_text = f"Frame #{idx + 1} (timestamp: {ts}s):"
        for t_name, res in tool_results.items():
            if "summary" in res:
                obs_text += f"\n  - Tool [{res.get('tool', t_name)}]: {res['summary']}"
        tool_summaries.append(obs_text)

        contents.append(f"=== Frame #{idx + 1} at timestamp {ts}s ===")
        contents.append(pil_img)

    # add the audio results too if the video has audio
    if audio_details and audio_details.get("has_audio"):
        audio_text = (
            f"AUDIO / ACOUSTIC FORENSIC FINDINGS:\n"
            f"- Spectral Cutoff Check: {audio_details.get('spectral_summary', 'N/A')}\n"
            f"- Voice Cadence / Volume Dynamic: {audio_details.get('cadence_summary', 'N/A')}\n"
            f"- Lip-Sync Correlation: {audio_details.get('lipsync_summary', 'N/A')}"
        )
        tool_summaries.append(audio_text)

    system_prompt = (
        "You are an expert video forensics analyst operating in a ReAct multi-agent framework.\n"
        "Examine the attached chronological video keyframes to detect whether this video is AUTHENTIC or AI-GENERATED/MANIPULATED.\n\n"
        "FORENSIC DETECTION CRITERIA:\n"
        "1. Generative AI Video (Sora, Kling, Runway Gen-3, Luma, Pika, Stable Video):\n"
        "   - Waxy, overly smooth, or plastic skin texture lacking natural pores and sensor grain.\n"
        "   - Warping, morphing, or drifting background architecture, text gibberish, or impossible geometry.\n"
        "   - Anatomical glitches: unnatural hands/fingers, distorted teeth, asymmetric pupils, melting limbs.\n"
        "   - Physics anomalies: unnatural fluid/smoke motion, inconsistent shadows, or gravity violations.\n"
        "2. Face-Swap / Deepfake Manipulation:\n"
        "   - Blurry boundary seams around the face oval, color/lighting mismatch between face and neck.\n"
        "   - Unnatural or frozen eye blinking, jittery landmark shifts across frames.\n"
        "3. Authentic Camera Footage:\n"
        "   - Natural optical motion blur adhering to camera shutter speed.\n"
        "   - Authentic ISO sensor noise grain, realistic human micro-expressions, consistent physical lighting.\n\n"
        "AUTOMATED FORENSIC TOOL OBSERVATIONS:\n" + "\n\n".join(tool_summaries) + "\n\n"
        "SCORING GUIDELINE:\n"
        "- 0.70 to 1.00: Clear AI generation or deepfake manipulation detected.\n"
        "- 0.40 to 0.65: Suspicious or ambiguous visual artifacts.\n"
        "- 0.00 to 0.25: Authentic real-world camera recording.\n"
    )

    if reflection_prompt:
        system_prompt += f"\nSelf-Correction Feedback: {reflection_prompt}\n"

    system_prompt += (
        "\nProvide your analysis as a JSON object with this exact structure:\n"
        "{\n"
        '  "overall_fake_score": <float between 0.00 (authentic) and 1.00 (manipulated)>,\n'
        '  "summary_reasoning": "<2-3 sentence overview explaining the forensic evidence for your verdict>",\n'
        '  "frame_explanations": {\n'
        '     "<timestamp_as_string>": "<1-2 sentence forensic observation for this specific frame>"\n'
        "  }\n"
        "}"
    )

    contents.insert(0, system_prompt)

    try:
        # gemini-3.5-flash-lite is fast and accurate enough for this, takes about 2s
        gen_config = types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0.1,
            max_output_tokens=1000
        )

        response = client.models.generate_content(
            model="gemini-3.5-flash-lite",
            contents=contents,
            config=gen_config
        )

        meta = getattr(response, "usage_metadata", None)
        _record_usage(usage, "gemini",
                      getattr(meta, "prompt_token_count", 0),
                      (getattr(meta, "candidates_token_count", 0) or 0) + (getattr(meta, "thoughts_token_count", 0) or 0))

        data = json.loads(response.text)
        overall_score = _clamp_score(data.get("overall_fake_score", 0.1))
        summary_reasoning = data.get("summary_reasoning", "Gemini analysis completed.")
        # gemini sometimes writes "1.333s", strip the s so it matches the groq format and the UI can find the frame
        frame_explanations = {str(k).strip().rstrip("s").strip(): v for k, v in (data.get("frame_explanations") or {}).items()}

        tools_list = sorted(list(all_tools_used))
        reasoning = f"Gemini 3.5 Flash-Lite batch-analyzed {len(suspicious_frames)} frames (Tools: {', '.join(tools_list)}): {summary_reasoning}"

        return reasoning, frame_explanations, round(overall_score, 4), tools_list

    except Exception as e:
        logger.error(f"Gemini API error: {e}")
        # returning None makes it fall back to groq
        return None


def process_single_groq_frame(f, client, all_frames, metadata, reflection_prompt, usage=None):
    # runs one frame through groq, gets called from the thread pool
    ts = round(f["timestamp"], 3)
    img_array = f["image"]

    tool_results, tools_called = run_tools_for_frame(f, all_frames=all_frames, metadata=metadata)

    # resize to 384 so the base64 image is smaller and uploads faster
    rgb = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    small_rgb = cv2.resize(rgb, (384, max(1, int(h * (384 / w)))))
    pil_img = Image.fromarray(small_rgb)

    buffered = io.BytesIO()
    pil_img.save(buffered, format="JPEG", quality=80)
    base64_image = base64.b64encode(buffered.getvalue()).decode('utf-8')

    tool_obs_text = ""
    for tool_key, res in tool_results.items():
        if "summary" in res:
            tool_obs_text += f"\n- Tool [{res.get('tool', tool_key)}]: {res['summary']}"

    prompt = (
        "You are a forensic video expert operating in a ReAct multi-agent framework.\n"
        "Analyze this frame for AI generation or deepfake manipulation.\n"
        f"Automated Tool Observations:{tool_obs_text}\n"
    )
    if reflection_prompt:
        prompt += f"\n{reflection_prompt}\n"
    prompt += (
        "\nCRITICAL requirement: Start your response with: [SCORE: X.XX] "
        "(0.00 is authentic, 1.00 is fully AI-generated/fake). Then, give a 2-sentence explanation."
    )

    # None means the call failed, so we leave it out of the average instead of counting it as authentic
    score = None
    explanation = "Analysis failed for this frame."

    try:
        response = client.chat.completions.create(
            model=GROQ_VISION_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
                    ],
                }
            ],
        )
        groq_usage = getattr(response, "usage", None)
        _record_usage(usage, "groq", getattr(groq_usage, "prompt_tokens", 0), getattr(groq_usage, "completion_tokens", 0))
        response_text = response.choices[0].message.content or ""
        # reasoning models can put a <think>...</think> block first, we only want the answer
        response_text = re.sub(r"<think>.*?</think>", "", response_text, flags=re.DOTALL).strip()
        explanation = response_text

        score_pattern = r'\[?SCORE:\s*(\d*\.?\d+)(?:/1(?:\.0)?)?\]?'
        match = re.search(score_pattern, response_text, re.IGNORECASE)
        if match:
            score = _clamp_score(match.group(1))
            explanation = re.sub(score_pattern, '', response_text, count=1, flags=re.IGNORECASE).strip()
        else:
            logger.warning(f"Groq response for frame t={ts}s had no parsable [SCORE]; excluding from aggregate")
    except Exception as err:
        logger.error(f"Groq API error on frame t={ts}s: {err}")

    return str(ts), explanation, score, tools_called


def analyze_with_groq(suspicious_frames, reflection_prompt="", metadata=None, all_frames=None, api_key=None, usage=None):
    # fallback: sends frames to Groq in parallel threads so we dont wait 10+ seconds sequentially
    from groq import Groq
    # default is 60s with 2 retries which can be 3 min per frame, so make it shorter
    client = Groq(api_key=api_key, timeout=30, max_retries=1)
    frame_explanations = {}
    scores_list = []
    all_tools_used = set()

    workers = min(len(suspicious_frames), 4)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(
            lambda f: process_single_groq_frame(f, client, all_frames, metadata, reflection_prompt, usage),
            suspicious_frames
        ))

    for ts_str, explanation, score, tools_called in results:
        frame_explanations[ts_str] = explanation
        if score is not None:
            scores_list.append(score)
        all_tools_used.update(tools_called)

    if not scores_list:
        raise RuntimeError(f"Groq API failed on all {len(suspicious_frames)} frames")

    sorted_scores = sorted(scores_list, reverse=True)
    top_scores = sorted_scores[:3]
    llm_score = sum(top_scores) / len(top_scores)

    tools_used_list = sorted(list(all_tools_used))
    llm_reasoning = f"Groq parallel-analyzed {len(suspicious_frames)} frames (Tools: {', '.join(tools_used_list)}). Top confidence: {round(llm_score, 2)}"
    return llm_reasoning, frame_explanations, round(llm_score, 4), tools_used_list


def analyze_with_llm(suspicious_frames, reflection_prompt="", metadata=None, all_frames=None, audio_details=None, usage=None):
    # main function, tries gemini first and then groq
    if not suspicious_frames:
        return "No suspicious frames flagged for analysis", {}, 0.0, []

    # check for the gemini key first
    gemini_key = os.environ.get("GEMINI_API_KEY")
    if gemini_key:
        try:
            logger.info("Running visual reasoning with Google Gemini 3.5 Flash-Lite...")
            res = analyze_with_gemini(
                suspicious_frames,
                reflection_prompt=reflection_prompt,
                metadata=metadata,
                all_frames=all_frames,
                api_key=gemini_key,
                audio_details=audio_details,
                usage=usage
            )
            if res:
                return res
        except Exception as e:
            logger.warning(f"Gemini failed, trying Groq fallback: {e}")

    # otherwise use groq
    groq_key = os.environ.get("GROQ_API_KEY")
    if groq_key:
        logger.info(f"Running visual reasoning with Groq ({GROQ_VISION_MODEL})...")
        return analyze_with_groq(
            suspicious_frames,
            reflection_prompt=reflection_prompt,
            metadata=metadata,
            all_frames=all_frames,
            api_key=groq_key,
            usage=usage
        )

    raise LLMUnavailableError("LLM analysis skipped because no GEMINI_API_KEY or GROQ_API_KEY is configured")
