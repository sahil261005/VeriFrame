import os
import io
import base64
import json
import re
import logging
from PIL import Image
import cv2
from concurrent.futures import ThreadPoolExecutor
from agents.tools import TOOL_REGISTRY

logger = logging.getLogger(__name__)


def pick_suspicious_frames(visual_flagged, temporal_flagged, all_frames, max_count=8):
    # grab the most suspicious frames from visual + temporal results
    # and backfill with evenly spaced frames if we dont have enough
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
    # run our forensic tools on a single frame and collect results
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


def analyze_with_gemini(suspicious_frames, reflection_prompt="", metadata=None, all_frames=None, api_key=None):
    # sends frames to Gemini 3.5 Flash-Lite for multi-image analysis and gets back a JSON verdict
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)
    contents = []
    tool_summaries = []
    all_tools_used = set()

    # run tools on all frames at once using threads (they are independent so this is safe)
    from concurrent.futures import ThreadPoolExecutor
    workers = min(len(suspicious_frames), 6)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        tool_results_list = list(pool.map(
            lambda f: run_tools_for_frame(f, all_frames=all_frames, metadata=metadata),
            suspicious_frames
        ))

    for idx, (f, (tool_results, tools_called)) in enumerate(zip(suspicious_frames, tool_results_list)):
        ts = round(f["timestamp"], 3)
        all_tools_used.update(tools_called)

        # Convert frame to compact PIL thumbnail for fast upload and instant tokenization
        rgb = cv2.cvtColor(f["image"], cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        target_w = 256
        target_h = max(1, int(h * (target_w / w)))
        small_rgb = cv2.resize(rgb, (target_w, target_h))
        pil_img = Image.fromarray(small_rgb)

        obs_text = f"Frame #{idx + 1} (timestamp: {ts}s):"
        for t_name, res in tool_results.items():
            if "summary" in res:
                obs_text += f"\n  - Tool [{res.get('tool', t_name)}]: {res['summary']}"
        tool_summaries.append(obs_text)

        contents.append(f"=== Frame #{idx + 1} at timestamp {ts}s ===")
        contents.append(pil_img)

    system_prompt = (
        "You are an expert video forensics analyst operating in a ReAct multi-agent framework.\n"
        "Examine the attached chronological video keyframes for AI generation (Sora, Runway, Pika, Kling) or deepfake face manipulation.\n\n"
        "Forensic Tool Observations:\n" + "\n\n".join(tool_summaries) + "\n\n"
    )

    if reflection_prompt:
        system_prompt += f"\n{reflection_prompt}\n"

    system_prompt += (
        "\nProvide your analysis as a JSON object with this exact structure:\n"
        "{\n"
        '  "overall_fake_score": <float between 0.0 (authentic) and 1.0 (manipulated)>,\n'
        '  "summary_reasoning": "<2-3 sentence overview explaining why the video is authentic or manipulated>",\n'
        '  "frame_explanations": {\n'
        '     "<timestamp_as_string>": "<1-2 sentence forensic observation for this specific frame>"\n'
        "  }\n"
        "}"
    )

    contents.insert(0, system_prompt)

    try:
        # use gemini-3.5-flash-lite for instant responses (~2-3 seconds)
        gen_config = types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0.1
        )

        response = client.models.generate_content(
            model="gemini-3.5-flash-lite",
            contents=contents,
            config=gen_config
        )

        data = json.loads(response.text)
        overall_score = float(data.get("overall_fake_score", 0.1))
        summary_reasoning = data.get("summary_reasoning", "Gemini analysis completed.")
        frame_explanations = data.get("frame_explanations", {})

        tools_list = sorted(list(all_tools_used))
        reasoning = f"Gemini 3.5 Flash-Lite batch-analyzed {len(suspicious_frames)} frames (Tools: {', '.join(tools_list)}): {summary_reasoning}"

        return reasoning, frame_explanations, round(overall_score, 4), tools_list

    except Exception as e:
        logger.error(f"Gemini API error: {e}")
        # fallback to Groq if Gemini throws an error
        return None


def analyze_with_groq(suspicious_frames, reflection_prompt="", metadata=None, all_frames=None, api_key=None):
    # fallback: sends frames one by one to Groq Llama if Gemini is unavailable
    from groq import Groq
    client = Groq(api_key=api_key)
    frame_explanations = {}
    scores_list = []
    num_analyzed = 0
    all_tools_used = set()

    for f in suspicious_frames:
        ts = round(f["timestamp"], 3)
        img_array = f["image"]

        tool_results, tools_called = run_tools_for_frame(f, all_frames=all_frames, metadata=metadata)
        all_tools_used.update(tools_called)

        rgb = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)

        buffered = io.BytesIO()
        pil_img.save(buffered, format="JPEG")
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

        try:
            response = client.chat.completions.create(
                model="meta-llama/llama-4-scout-17b-16e-instruct",
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
            response_text = response.choices[0].message.content or ""
            score = 0.1
            explanation = response_text

            match = re.search(r'\[?SCORE:\s*([0-9.]+)(?:/1\.0)?\]?', response_text, re.IGNORECASE)
            if match:
                try:
                    score = float(match.group(1))
                    explanation = re.sub(r'\[?SCORE:\s*[0-9.]+(?:/1\.0)?\]?', '', response_text, flags=re.IGNORECASE).strip()
                except Exception:
                    pass

            ts_str = str(ts)
            frame_explanations[ts_str] = explanation
            scores_list.append(score)
            num_analyzed += 1
        except Exception as err:
            logger.error(f"Groq API error on frame t={ts}s: {err}")

    if num_analyzed > 0:
        sorted_scores = sorted(scores_list, reverse=True)
        top_scores = sorted_scores[:3]
        llm_score = sum(top_scores) / len(top_scores)
    else:
        llm_score = 0.0

    tools_used_list = sorted(list(all_tools_used))
    llm_reasoning = f"Groq analyzed {num_analyzed} frames (Tools: {', '.join(tools_used_list)}). Top confidence: {round(llm_score, 2)}"
    return llm_reasoning, frame_explanations, round(llm_score, 4), tools_used_list


def analyze_with_llm(suspicious_frames, reflection_prompt="", metadata=None, all_frames=None):
    # main entry point: tries Gemini first, falls back to Groq
    if not suspicious_frames:
        return "No suspicious frames flagged for analysis", {}, 0.0, []

    # Check for Gemini API key first
    gemini_key = os.environ.get("GEMINI_API_KEY")
    if gemini_key:
        try:
            logger.info("Running visual reasoning with Google Gemini 3.5 Flash-Lite...")
            res = analyze_with_gemini(
                suspicious_frames,
                reflection_prompt=reflection_prompt,
                metadata=metadata,
                all_frames=all_frames,
                api_key=gemini_key
            )
            if res:
                return res
        except Exception as e:
            logger.warning(f"Gemini failed, trying Groq fallback: {e}")

    # Fallback to Groq
    groq_key = os.environ.get("GROQ_API_KEY")
    if groq_key:
        logger.info("Running visual reasoning with Groq Llama 3.2 Vision...")
        return analyze_with_groq(
            suspicious_frames,
            reflection_prompt=reflection_prompt,
            metadata=metadata,
            all_frames=all_frames,
            api_key=groq_key
        )

    # If neither key is provided
    frame_explanations = {str(round(f["timestamp"], 3)): "Analysis skipped (no GEMINI_API_KEY or GROQ_API_KEY set)." for f in suspicious_frames}
    return "LLM analysis skipped because no API key is configured", frame_explanations, 0.0, []
