import re
import logging

logger = logging.getLogger(__name__)


def reflect_on_analysis(llm_score, frame_explanations, llm_reasoning):
    """
    looks over what the llm said and checks if it contradicts itself.
    1. does the score match what the text says
    2. do the frame explanations disagree with each other
    3. are any frame explanations empty or missing
    returns a dict with needs_correction and correction_prompt
    """
    issues_found = []

    # check 1: does the score match the reasoning
    # score above 0.7 but the text says "authentic" or "real" is a contradiction
    # score below 0.3 but the text says "fake" or "manipulated" is also one
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

    # check 2: do frames disagree with each other
    # like one frame says the lighting is natural and another says its artificial
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

    # check 3: missing or empty explanations
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

    # if we found problems, build a prompt telling the llm what to fix
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
