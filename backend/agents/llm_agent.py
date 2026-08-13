import os
import io
import base64
import re
import logging
from PIL import Image
from groq import Groq
from agents.tools import TOOL_REGISTRY

logger = logging.getLogger(__name__)


def pick_suspicious_frames(visual_flagged, temporal_flagged, all_frames, max_count=8):
    """
    merges timestamps flagged by visual and temporal agents, sorts them,
    and returns up to max_count matching frames from all_frames.
    max_count is dynamic (8 by default, or 12 if router chose extended mode).
    """
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
                
    # if not enough flagged frames, backfill with evenly spaced frames
    if len(suspicious_frames) < max_count and len(all_frames) > 0:
        existing_ts = {round(f["timestamp"], 3) for f in suspicious_frames}
        step = max(1, len(all_frames) // max_count)
        for i in range(0, len(all_frames), step):
            f = all_frames[i]
            ts = round(f["timestamp"], 3)
            if ts not in existing_ts:
                suspicious_frames.append(f)
                existing_ts.add(ts)
            if len(suspicious_frames) >= max_count:
                break

    return suspicious_frames[:max_count]


def run_tools_for_frame(frame_data, all_frames=None, metadata=None):
    """
    ReAct tool execution step: runs applicable forensic tools on a frame
    to produce deterministic tool observations before calling LLM.
    """
    tool_results = {}
    tools_called = []

    # 1. noise & edge tool
    try:
        res = TOOL_REGISTRY["analyze_noise_pattern"]["function"](frame_data)
        tool_results["noise_pattern"] = res
        tools_called.append("analyze_noise_pattern")
    except Exception as e:
        logger.warning(f"tool analyze_noise_pattern failed: {e}")

    # 2. face landmarks tool
    try:
        res = TOOL_REGISTRY["check_face_landmarks"]["function"](frame_data)
        tool_results["face_landmarks"] = res
        tools_called.append("check_face_landmarks")
    except Exception as e:
        logger.warning(f"tool check_face_landmarks failed: {e}")

    # 3. optical flow comparison tool (if all_frames provided)
    if all_frames:
        try:
            res = TOOL_REGISTRY["compare_adjacent_frames"]["function"](frame_data, all_frames)
            tool_results["adjacent_frames"] = res
            tools_called.append("compare_adjacent_frames")
        except Exception as e:
            logger.warning(f"tool compare_adjacent_frames failed: {e}")

    # 4. metadata check tool (if metadata provided)
    if metadata and "metadata" not in tool_results:
        try:
            res = TOOL_REGISTRY["check_metadata"]["function"](metadata)
            tool_results["metadata"] = res
            tools_called.append("check_metadata")
        except Exception as e:
            logger.warning(f"tool check_metadata failed: {e}")

    return tool_results, tools_called


def analyze_with_llm(suspicious_frames, reflection_prompt="", metadata=None, all_frames=None):
    """
    analyzes suspicious frames using Llama 3.2 Vision on Groq,
    enriched with ReAct tool observations and handling reflection feedback.
    """
    if not suspicious_frames:
        return "No suspicious frames flagged for analysis", {}, 0.0, []

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        frame_explanations = {}
        for f in suspicious_frames:
            ts_str = str(round(f["timestamp"], 3))
            frame_explanations[ts_str] = "Analysis skipped (no GROQ_API_KEY set)."
        return "Groq analysis skipped because GROQ_API_KEY is missing", frame_explanations, 0.0, []

    client = Groq(api_key=api_key)
    frame_explanations = {}
    scores_list = []
    num_analyzed = 0
    all_tools_used = set()

    for f in suspicious_frames:
        ts = round(f["timestamp"], 3)
        img_array = f["image"]

        # Run ReAct tools first to gather empirical evidence for prompt context
        tool_results, tools_called = run_tools_for_frame(f, all_frames=all_frames, metadata=metadata)
        all_tools_used.update(tools_called)

        # Convert BGR to RGB and PIL Image
        rgb = img_array[:, :, ::-1]
        pil_img = Image.fromarray(rgb)

        # Convert PIL Image to base64 jpeg
        buffered = io.BytesIO()
        pil_img.save(buffered, format="JPEG")
        base64_image = base64.b64encode(buffered.getvalue()).decode('utf-8')

        # Build prompt incorporating tool observations and reflection feedback
        tool_obs_text = ""
        for tool_key, res in tool_results.items():
            if "summary" in res:
                tool_obs_text += f"\n- Tool [{res.get('tool', tool_key)}]: {res['summary']}"

        prompt = (
            "You are a forensic video expert operating in a ReAct multi-agent framework.\n"
            "Analyze this frame from a video for AI generation (e.g. Sora, Runway, Pika, Synthesia) or deepfake manipulation.\n"
            f"Automated Forensic Tool Observations:{tool_obs_text}\n\n"
            "Look closely for: structural distortions, physics violations, temporal morphing artifacts, lighting/shadow mismatches, flat/plasticky textures, or facial inconsistencies.\n"
        )

        if reflection_prompt:
            prompt += f"\n{reflection_prompt}\n"

        prompt += (
            "CRITICAL requirement: You MUST start your response with a rating in this exact format: [SCORE: X.XX] "
            "(e.g., [SCORE: 0.90]) where X.XX is your confidence that this frame is synthetic/AI-generated or manipulated "
            "(0.00 is authentic camera footage, 1.00 is fully AI-generated/fake). Then, provide your maximum 2-sentence explanation."
        )

        try:
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

            # Parse score via regex
            match = re.search(r'\[?SCORE:\s*([0-9.]+)(?:/1\.0)?\]?', response_text, re.IGNORECASE)
            if match:
                try:
                    score = float(match.group(1))
                    explanation = re.sub(r'\[?SCORE:\s*[0-9.]+(?:/1\.0)?\]?', '', response_text, flags=re.IGNORECASE).strip()
                except Exception:
                    pass
            else:
                numbers = re.findall(r'\b0\.[0-9]+\b|\b1\.0\b', response_text)
                if numbers:
                    try:
                        score = float(numbers[0])
                    except Exception:
                        pass
                else:
                    text_lower = response_text.lower()
                    if any(word in text_lower for word in ["manipulated", "fake", "ai-generated", "synthetic", "deepfake", "distortions"]):
                        score = 0.85
                    elif "authentic" in text_lower or "real" in text_lower:
                        score = 0.05

            ts_str = str(ts)
            frame_explanations[ts_str] = explanation
            scores_list.append(score)
            num_analyzed += 1

        except Exception as err:
            logger.error(f"Groq API error on frame t={ts}s: {err}")
            ts_str = str(ts)
            frame_explanations[ts_str] = f"Frame analysis encountered error: {err}"

    # Average top 3 highest scores
    if num_analyzed > 0:
        sorted_scores = sorted(scores_list, reverse=True)
        top_scores = sorted_scores[:3]
        llm_score = sum(top_scores) / len(top_scores)
    else:
        llm_score = 0.0

    tools_used_list = sorted(list(all_tools_used))
    llm_reasoning = (
        f"ReAct agent analyzed {num_analyzed} frames using tools: {', '.join(tools_used_list) if tools_used_list else 'none'}. "
        f"Top 3 frame average confidence: {round(llm_score, 2)}"
    )

    return llm_reasoning, frame_explanations, round(llm_score, 4), tools_used_list
