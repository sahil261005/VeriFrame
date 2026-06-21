def compute_verdict(visual_score, temporal_score, llm_score, agent_status):
    # calculate final verdict from visual, temporal, and llm scores. redistribute weights if any node fails
    base_weights = {
        "visual": 0.40,
        "temporal": 0.30,
        "llm": 0.30
    }
    
    scores = {
        "visual": visual_score,
        "temporal": temporal_score,
        "llm": llm_score
    }

    # check which agents succeeded and which failed
    active_weights = {}
    total_active_base_weight = 0.0

    for name, weight in base_weights.items():
        # if the agent status is not "failed", we use its score
        if agent_status.get(name) != "failed":
            active_weights[name] = weight
            total_active_base_weight += weight

    # fallback just in case everything failed
    if total_active_base_weight == 0.0:
        for name in base_weights:
            active_weights[name] = 1.0 / len(base_weights)
        total_active_base_weight = 1.0

    # calculate normalized weights (so they add up to 1.0)
    normalized_weights = {}
    for name, weight in active_weights.items():
        normalized_weights[name] = weight / total_active_base_weight

    # compute final consensus score
    final_confidence = 0.0
    for name, weight in normalized_weights.items():
        final_confidence += scores[name] * weight

    # determine final verdict based on score thresholds
    if final_confidence > 0.70:
        verdict = "MANIPULATED"
    elif final_confidence < 0.30:
        verdict = "AUTHENTIC"
    else:
        verdict = "UNCERTAIN"

    return verdict, round(final_confidence, 4), normalized_weights


def build_report(state):
    # compile all agent metrics and metadata into a final dictionary
    # check if any agent failed
    agent_status = state.get("agent_status", {})
    is_partial = False
    for status in agent_status.values():
        if status == "failed":
            is_partial = True
            break

    # compile all findings into a clean report dict
    report = {
        "verdict": state.get("final_verdict", "UNCERTAIN"),
        "confidence": state.get("final_confidence", 0.0),
        "is_partial_analysis": is_partial,
        "video_metadata": state.get("metadata", {}),
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
            "llm_agent": {
                "score": state.get("llm_score", 0.0),
                "status": agent_status.get("llm", "unknown"),
                "reasoning": state.get("llm_reasoning", "")
            }
        },
        "frame_level_details": state.get("frame_explanations", {})
    }

    return report
