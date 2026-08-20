def compute_verdict(visual_score, temporal_score, llm_score, agent_status, metadata=None, audio_score=0.0, has_audio=False):
    # base weights across active forensic modalities
    if has_audio and agent_status.get("audio") == "success":
        base_weights = {
            "visual": 0.35,
            "temporal": 0.15,
            "llm": 0.35,
            "audio": 0.15
        }
    else:
        base_weights = {
            "visual": 0.45,
            "temporal": 0.15,
            "llm": 0.40
        }

    if llm_score > 0.60:
        if has_audio and agent_status.get("audio") == "success":
            base_weights = {
                "visual": 0.15,
                "temporal": 0.10,
                "llm": 0.60,
                "audio": 0.15
            }
        else:
            base_weights = {
                "visual": 0.20,
                "temporal": 0.10,
                "llm": 0.70
            }
    elif agent_status.get("visual") == "fallback":
        if has_audio and agent_status.get("audio") == "success":
            base_weights = {
                "visual": 0.10,
                "temporal": 0.15,
                "llm": 0.60,
                "audio": 0.15
            }
        else:
            base_weights = {
                "visual": 0.10,
                "temporal": 0.15,
                "llm": 0.75
            }
    
    scores = {
        "visual": visual_score,
        "temporal": temporal_score,
        "llm": llm_score,
        "audio": audio_score
    }

    active_weights = {}
    total_active_base_weight = 0.0

    for name, weight in base_weights.items():
        if agent_status.get(name) != "failed":
            active_weights[name] = weight
            total_active_base_weight += weight

    if total_active_base_weight == 0.0:
        for name in base_weights:
            active_weights[name] = 1.0 / len(base_weights)
        total_active_base_weight = 1.0

    normalized_weights = {}
    for name, weight in active_weights.items():
        normalized_weights[name] = weight / total_active_base_weight

    final_confidence = 0.0
    for name, weight in normalized_weights.items():
        final_confidence += scores[name] * weight

    # Domain-specific detector alignment:
    # 1. ViT model specializes in facial deepfake boundary artifacts (FaceForensics)
    # 2. Gemini LLM Vision specializes in modern generative AI diffusion (Sora/Kling/Runway)
    # If either specialized detector is confident (>= 0.55), elevate confidence to reflect detection.
    max_specialized = max(visual_score, llm_score)
    if max_specialized >= 0.55:
        final_confidence = max(final_confidence, max_specialized * 0.85)

    if metadata:
        robustness = metadata.get("robustness_score", 1.0)
        if robustness < 0.8:
            diff = final_confidence - 0.5
            final_confidence = 0.5 + diff * robustness

    if final_confidence >= 0.48:
        verdict = "MANIPULATED"
    elif final_confidence < 0.30:
        verdict = "AUTHENTIC"
    else:
        verdict = "UNCERTAIN"

    return verdict, round(final_confidence, 4), normalized_weights


def build_report(state):
    agent_status = state.get("agent_status", {})
    is_partial = False
    for status in agent_status.values():
        if status == "failed":
            is_partial = True
            break

    metadata = state.get("metadata", {})
    provenance = metadata.get("provenance", {})

    report = {
        "verdict": state.get("final_verdict", "UNCERTAIN"),
        "confidence": state.get("final_confidence", 0.0),
        "is_partial_analysis": is_partial,
        "video_metadata": metadata,
        "agent_breakdown": {
            "visual_agent": {
                "score": state.get("visual_score", 0.0),
                "status": agent_status.get("visual", "unknown"),
                "flagged_frames_count": len(state.get("visual_flagged_frames", []))
            },
            "temporal_agent": {
                "score": state.get("temporal_score", 0.0),
                "status": agent_status.get("temporal", "unknown"),
                "flow_anomalies_count": sum(1 for r in state.get("flow_results", []) if r.get("is_anomalous")),
                "face_inconsistencies_count": sum(1 for r in state.get("face_results", []) if r.get("is_inconsistent"))
            },
            "audio_agent": {
                "score": state.get("audio_score", 0.0),
                "status": agent_status.get("audio", "unknown"),
                "has_audio": state.get("has_audio", False),
                "details": state.get("audio_details", {})
            },
            "llm_agent": {
                "score": state.get("llm_score", 0.0),
                "status": agent_status.get("llm", "unknown"),
                "reasoning": state.get("llm_reasoning", "")
            },
            "provenance_agent": {
                "score": provenance.get("provenance_score", 0.5),
                "status": "success",
                "c2pa_compliant": provenance.get("c2pa_compliant", False),
                "encoder": provenance.get("encoder", "unknown"),
                "metadata_stripped": provenance.get("metadata_stripped", True),
                "is_camera_filename": provenance.get("is_camera_filename", False),
                "is_social_filename": provenance.get("is_social_filename", False)
            }
        },
        "frame_level_details": state.get("frame_explanations", {}),
        "visual_per_frame": state.get("visual_per_frame", [])
    }

    return report
