def compute_verdict(visual_score, temporal_score, llm_score, agent_status, metadata=None, audio_score=0.0, has_audio=False, generative_score=0.0):
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
    
    # the generative-video detector is the agent aimed at fully AI-generated footage (Sora/Kling/Runway/Veo),
    # which the face-swap visual model cannot see, so it takes the largest single share when it ran
    if agent_status.get("generative") == "success":
        base_weights = {name: w * 0.55 for name, w in base_weights.items()}
        base_weights["generative"] = 0.45

    scores = {
        "generative": generative_score,
        "visual": visual_score,
        "temporal": temporal_score,
        "llm": llm_score,
        "audio": audio_score
    }

    active_weights = {}
    total_active_base_weight = 0.0

    for name, weight in base_weights.items():
        if agent_status.get(name) not in ("failed", "skipped"):
            active_weights[name] = weight
            total_active_base_weight += weight

    # no agent produced usable evidence: averaging zeros would wrongly report AUTHENTIC
    if total_active_base_weight == 0.0:
        return "UNCERTAIN", 0.0, {}

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
    # only trust a specialist's override when it actually ran its model (not heuristics/failed/skipped)
    specialized = [score for name, score in (("visual", visual_score), ("llm", llm_score), ("generative", generative_score))
                   if agent_status.get(name) == "success"]
    max_specialized = max(specialized, default=0.0)
    if max_specialized >= 0.55:
        final_confidence = max(final_confidence, max_specialized * 0.85)

    if metadata:
        robustness = metadata.get("robustness_score", 1.0)
        if robustness < 0.8:
            diff = final_confidence - 0.5
            final_confidence = 0.5 + diff * robustness

    # the file's own signed Content Credentials say a generative-AI tool produced it: that is direct evidence,
    # stronger than any pixel-level detector, so it decides the verdict
    if metadata and metadata.get("provenance", {}).get("c2pa_ai_generated"):
        final_confidence = max(final_confidence, 0.95)

    if final_confidence >= 0.48:
        verdict = "MANIPULATED"
    elif final_confidence < 0.30:
        verdict = "AUTHENTIC"
    else:
        verdict = "UNCERTAIN"

    # "AUTHENTIC" needs positive evidence against fully AI-generated footage, which the face-swap and motion
    # checks cannot see: either the LLM ran, or the generative detector ran and found nothing at all
    if verdict == "AUTHENTIC":
        llm_ok = agent_status.get("llm") == "success"
        generative_clear = agent_status.get("generative") == "success" and generative_score < 0.15
        if not (llm_ok or generative_clear):
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
            "generative_agent": {
                "score": state.get("generative_score", 0.0),
                "status": agent_status.get("generative", "unknown"),
                "flagged_frames_count": sum(1 for r in state.get("generative_per_frame", []) if (r.get("generated_probability") or 0) > 0.5)
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
                "is_social_filename": provenance.get("is_social_filename", False),
                "c2pa_present": provenance.get("c2pa_present", False),
                "c2pa_signature_valid": provenance.get("c2pa_signature_valid", False),
                "c2pa_trusted_signer": provenance.get("c2pa_trusted_signer", False),
                "c2pa_ai_generated": provenance.get("c2pa_ai_generated", False),
                "c2pa_generator": provenance.get("c2pa_generator")
            }
        },
        "frame_level_details": state.get("frame_explanations", {}),
        "visual_per_frame": state.get("visual_per_frame", [])
    }

    return report
