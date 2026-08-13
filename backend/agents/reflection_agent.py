import re
import logging

logger = logging.getLogger(__name__)


def reflect_on_analysis(llm_score, frame_explanations, llm_reasoning):
    """
    reviews the LLM agent's output for internal contradictions and score-reasoning misalignment.
    
    checks performed:
    1. score-reasoning alignment: does the numerical score match the text sentiment?
    2. cross-frame consistency: do frame explanations contradict each other?
    3. coverage check: are there frames with empty or missing explanations?
    
    returns a dict with needs_correction (bool) and correction_prompt (str).
    """
    issues_found = []

    # check 1: score-reasoning alignment
    # if score is high (> 0.7) but reasoning text says "authentic" or "real", thats contradictory
    # if score is low (< 0.3) but reasoning says "fake" or "manipulated", also contradictory
    all_explanation_text = " ".join(frame_explanations.values()).lower()

    authentic_words = ["authentic", "genuine", "real camera", "no manipulation", "appears real", "naturally captured"]
    fake_words = ["manipulated", "synthetic", "ai-generated", "deepfake", "artificially", "generated content", "fake"]

    has_authentic_language = any(word in all_explanation_text for word in authentic_words)
    has_fake_language = any(word in all_explanation_text for word in fake_words)

    if llm_score > 0.70 and has_authentic_language and not has_fake_language:
        issues_found.append(
            f"Score is high ({llm_score:.2f}) suggesting manipulation, but frame explanations use "
            f"language like 'authentic' or 'real'. The score and reasoning are contradictory."
        )

    if llm_score < 0.30 and has_fake_language and not has_authentic_language:
        issues_found.append(
            f"Score is low ({llm_score:.2f}) suggesting authentic content, but frame explanations "
            f"mention 'manipulated', 'fake', or 'AI-generated'. The score and reasoning conflict."
        )

    # check 2: cross-frame consistency
    # look for frames where one says "lighting is natural" and another says "lighting is artificial"
    # on what appears to be the same subject
    lighting_natural = False
    lighting_artificial = False
    texture_natural = False
    texture_artificial = False

    for ts, explanation in frame_explanations.items():
        text = explanation.lower()
        if "natural lighting" in text or "consistent lighting" in text:
            lighting_natural = True
        if "artificial lighting" in text or "inconsistent lighting" in text or "lighting mismatch" in text:
            lighting_artificial = True
        if "natural skin" in text or "realistic texture" in text:
            texture_natural = True
        if "plasticky" in text or "synthetic texture" in text or "artificial skin" in text:
            texture_artificial = True

    if lighting_natural and lighting_artificial:
        issues_found.append(
            "Contradictory lighting assessments: some frames describe 'natural lighting' while "
            "others describe 'artificial/inconsistent lighting'. Re-examine the frames for a consistent assessment."
        )

    if texture_natural and texture_artificial:
        issues_found.append(
            "Contradictory texture assessments: some frames describe 'natural skin/texture' while "
            "others describe 'plasticky/synthetic texture'. Clarify whether the texture varies across "
            "regions or is consistently suspicious."
        )

    # check 3: missing/empty explanations
    empty_count = 0
    for ts, explanation in frame_explanations.items():
        if not explanation or len(explanation.strip()) < 10:
            empty_count += 1

    if empty_count > 0 and len(frame_explanations) > 0:
        ratio = empty_count / len(frame_explanations)
        if ratio > 0.3:
            issues_found.append(
                f"{empty_count} out of {len(frame_explanations)} frame explanations are missing or too short. "
                f"Provide detailed analysis for each frame."
            )

    # build the correction prompt if issues were found
    needs_correction = len(issues_found) > 0

    if needs_correction:
        correction_prompt = (
            "REFLECTION FEEDBACK: Your previous analysis had the following issues that need correction:\n"
        )
        for i, issue in enumerate(issues_found, 1):
            correction_prompt += f"{i}. {issue}\n"
        correction_prompt += (
            "\nPlease re-analyze the frames and provide corrected scores and explanations "
            "that are internally consistent. Make sure your SCORE aligns with your written reasoning."
        )
    else:
        correction_prompt = ""

    logger.info(f"reflection check: {len(issues_found)} issues found. needs_correction={needs_correction}")

    return {
        "needs_correction": needs_correction,
        "correction_prompt": correction_prompt,
        "issues_found": issues_found
    }
