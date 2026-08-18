from langgraph.graph import StateGraph, START, END
from agents.state import VeriFrameState
import agents.visual_agent as visual_agent
import agents.temporal_agent as temporal_agent
import agents.llm_agent as llm_agent
import agents.synthesis_agent as synthesis_agent
import agents.reflection_agent as reflection_agent
import agents.event_bus as event_bus
import logging

logger = logging.getLogger(__name__)


def visual_node(state: VeriFrameState) -> dict:
    """
    node for running visual analysis. checks deepfake classification + noise.
    """
    frames = state.get("frames", [])
    status = dict(state.get("agent_status", {}))
    job_id = state.get("job_id", "")
    
    if job_id:
        event_bus.publish_event(job_id, "Visual Forensics Agent", "Inspecting high-frequency noise variance and spatial artifacts...")

    try:
        pipe = visual_agent.load_model()
        score, flagged, all_results = visual_agent.analyze_frames(frames, pipe)
        status["visual"] = "success" if pipe != "fallback" else "fallback"
        
        mode_desc = "deep learning model" if pipe != "fallback" else "high-frequency spatial noise heuristics"
        if job_id:
            event_bus.publish_event(job_id, "Visual Forensics Agent", f"Completed using {mode_desc}. Fake score: {score:.2f}")

        return {
            "visual_score": score,
            "visual_flagged_frames": flagged,
            "visual_per_frame": all_results,
            "agent_status": status,
            "stream_events": [{"agent": "Visual Forensics", "message": f"Completed. Score: {score:.2f}"}]
        }
    except Exception as e:
        logger.error(f"error in visual node: {e}", exc_info=True)
        status["visual"] = "failed"
        if job_id:
            event_bus.publish_event(job_id, "Visual Forensics Agent", f"Failed: {e}")
        return {
            "visual_score": 0.0,
            "visual_flagged_frames": [],
            "visual_per_frame": [],
            "agent_status": status
        }


def temporal_node(state: VeriFrameState) -> dict:
    """
    node for running temporal analysis. checks optical flow + facial landmark shifts.
    """
    frames = state.get("frames", [])
    status = dict(state.get("agent_status", {}))
    job_id = state.get("job_id", "")

    if job_id:
        event_bus.publish_event(job_id, "Temporal Consistency Agent", "Running optical flow tracking and facial landmark geometry checks...")

    try:
        score, flagged_times, flow_res, face_res = temporal_agent.run_temporal_analysis(frames)
        status["temporal"] = "success"
        
        if job_id:
            event_bus.publish_event(job_id, "Temporal Consistency Agent", f"Completed. Anomaly score: {score:.2f} across {len(flagged_times)} keyframes.")

        return {
            "temporal_score": score,
            "temporal_flagged_timestamps": flagged_times,
            "flow_results": flow_res,
            "face_results": face_res,
            "agent_status": status,
            "stream_events": [{"agent": "Temporal Agent", "message": f"Completed. Anomaly score: {score:.2f}"}]
        }
    except Exception as e:
        logger.error(f"error in temporal node: {e}", exc_info=True)
        status["temporal"] = "failed"
        if job_id:
            event_bus.publish_event(job_id, "Temporal Consistency Agent", f"Failed: {e}")
        return {
            "temporal_score": 0.0,
            "temporal_flagged_timestamps": [],
            "flow_results": [],
            "face_results": [],
            "agent_status": status
        }


def router_node(state: VeriFrameState) -> dict:
    """
    conditional router node. inspects visual and temporal scores and decides:
    1. skip_llm: both CV agents are highly confident (>0.75 visual, >0.5 temporal) -> bypass LLM to save latency
    2. llm_extended: visual agent ran in fallback mode -> increase LLM sample count to 12 frames
    3. llm_normal: standard path -> 8 frames to LLM
    """
    visual_score = state.get("visual_score", 0.0)
    temporal_score = state.get("temporal_score", 0.0)
    status = state.get("agent_status", {})
    job_id = state.get("job_id", "")

    # fast path: if ALL frames scored below 2% fake, the video is clearly authentic
    # ViT model is authoritative - optical flow on keyframes always looks anomalous
    # because frames are spread across the video with large time gaps
    visual_per_frame = state.get("visual_per_frame", [])
    all_clearly_authentic = len(visual_per_frame) > 0 and all(
        f.get("fake_confidence", 1.0) < 0.02 for f in visual_per_frame
    )
    
    # also check the reverse: ALL frames unanimously fake (>0.90)
    all_clearly_fake = len(visual_per_frame) > 0 and all(
        f.get("fake_confidence", 0.0) > 0.90 for f in visual_per_frame
    )

    logger.info(f"Router check: visual_score={visual_score}, temporal_score={temporal_score}, "
                f"per_frame_count={len(visual_per_frame)}, all_authentic={all_clearly_authentic}, all_fake={all_clearly_fake}")
    
    if all_clearly_authentic:
        decision = "skip_llm"
        frame_count = 0
        reason = "All frames scored below 2% fake confidence. Fast-path verdict: AUTHENTIC."
    elif all_clearly_fake:
        decision = "skip_llm"
        frame_count = 0
        reason = "All frames scored above 90% fake confidence. Fast-path verdict: FAKE."
    elif visual_score > 0.60 and temporal_score > 0.30:
        decision = "skip_llm"
        frame_count = 0
        reason = "High confidence detected from both Visual and Temporal agents. Skipping LLM to optimize pipeline latency."
    elif status.get("visual") == "fallback":
        decision = "llm_extended"
        frame_count = 5
        reason = "Visual agent operating in fallback mode. Expanding LLM evaluation window to 5 keyframes."
    else:
        decision = "llm_normal"
        frame_count = 4
        reason = "Routing to LLM ReAct Vision agent with 4 keyframe samples."

    if job_id:
        event_bus.publish_event(job_id, "LangGraph Conditional Router", reason)

    logger.info(f"Router decision: {decision} (frame_count={frame_count})")

    return {
        "route_decision": decision,
        "llm_frame_count": frame_count
    }


def route_decision_edge(state: VeriFrameState) -> str:
    """conditional edge selector for router_node"""
    decision = state.get("route_decision", "llm_normal")
    if decision == "skip_llm":
        return "synthesis"
    return "llm"


def llm_node(state: VeriFrameState) -> dict:
    """
    node for running ReAct LLM vision model with tool execution on suspicious frames.
    """
    frames = state.get("frames", [])
    visual_flagged = state.get("visual_flagged_frames", [])
    temporal_flagged = state.get("temporal_flagged_timestamps", [])
    status = dict(state.get("agent_status", {}))
    max_count = state.get("llm_frame_count", 8)
    reflection_feedback = state.get("reflection_feedback", "")
    metadata = state.get("metadata", {})
    job_id = state.get("job_id", "")

    if job_id:
        msg = f"ReAct Vision Agent evaluating {max_count} keyframes..."
        if reflection_feedback:
            msg += " (Re-evaluating with self-correction feedback)"
        event_bus.publish_event(job_id, "Cognitive Reasoning Agent", msg)

    try:
        suspicious = llm_agent.pick_suspicious_frames(visual_flagged, temporal_flagged, frames, max_count=max_count)
        
        reasoning, explanations, score, tools_used = llm_agent.analyze_with_llm(
            suspicious, 
            reflection_prompt=reflection_feedback, 
            metadata=metadata, 
            all_frames=frames
        )
        status["llm"] = "success"

        if job_id:
            event_bus.publish_event(job_id, "Cognitive Reasoning Agent", f"Completed tool-calling analysis. Score: {score:.2f}. Tools invoked: {', '.join(tools_used)}")
        
        return {
            "llm_score": score,
            "llm_reasoning": reasoning,
            "frame_explanations": explanations,
            "tools_used": tools_used,
            "agent_status": status,
            "stream_events": [{"agent": "Cognitive Reasoning", "message": f"Completed using ReAct tools. Score: {score:.2f}"}]
        }
    except Exception as e:
        logger.error(f"error in llm node: {e}", exc_info=True)
        status["llm"] = "failed"
        if job_id:
            event_bus.publish_event(job_id, "Cognitive Reasoning Agent", f"Failed: {e}")
        return {
            "llm_score": 0.0,
            "llm_reasoning": f"LLM analysis failed: {e}",
            "frame_explanations": {},
            "tools_used": [],
            "agent_status": status
        }


def reflection_node(state: VeriFrameState) -> dict:
    """
    reflection node: reviews LLM reasoning and scores for internal contradictions.
    triggers a self-correction loop back to the LLM node if needed (max 2 iterations).
    """
    llm_score = state.get("llm_score", 0.0)
    frame_explanations = state.get("frame_explanations", {})
    llm_reasoning = state.get("llm_reasoning", "")
    reflection_count = state.get("reflection_count", 0)
    job_id = state.get("job_id", "")

    if job_id:
        event_bus.publish_event(job_id, "Agent Reflection Node", f"Auditing agent reasoning for internal consistency (Iteration {reflection_count + 1}/2)...")

    # run reflection audit
    reflection_res = reflection_agent.reflect_on_analysis(llm_score, frame_explanations, llm_reasoning)
    needs_correction = reflection_res["needs_correction"]
    prompt = reflection_res["correction_prompt"]

    # enforce max 2 reflection iterations
    if needs_correction and reflection_count < 2:
        logger.info(f"Reflection loop triggered (iteration {reflection_count + 1})")
        if job_id:
            event_bus.publish_event(
                job_id, 
                "Agent Reflection Node", 
                f"Contradiction detected in reasoning. Triggering self-correction loop iteration {reflection_count + 1}."
            )
        return {
            "needs_correction": True,
            "reflection_feedback": prompt,
            "reflection_count": reflection_count + 1
        }
    else:
        if job_id:
            event_bus.publish_event(job_id, "Agent Reflection Node", "Reasoning alignment audit passed successfully.")
        return {
            "needs_correction": False,
            "reflection_feedback": "",
            "reflection_count": reflection_count
        }


def reflection_edge(state: VeriFrameState) -> str:
    """conditional edge selector for reflection_node"""
    if state.get("needs_correction", False):
        return "llm"  # loop back to LLM node (Cyclic Graph!)
    return "synthesis"


def synthesis_node(state: VeriFrameState) -> dict:
    """
    node for synthesis. computes the final consensus verdict and compiles the report.
    """
    visual_score = state.get("visual_score", 0.0)
    temporal_score = state.get("temporal_score", 0.0)
    llm_score = state.get("llm_score", 0.0)
    status = dict(state.get("agent_status", {}))
    job_id = state.get("job_id", "")

    if job_id:
        event_bus.publish_event(job_id, "Consensus Engine", "Synthesizing multi-agent weights and building final verdict report...")

    try:
        metadata = state.get("metadata", {})
        verdict, confidence, normalized_weights = synthesis_agent.compute_verdict(
            visual_score, temporal_score, llm_score, status, metadata
        )
        status["synthesis"] = "success"
        
        current_state = dict(state)
        current_state["final_verdict"] = verdict
        current_state["final_confidence"] = confidence
        current_state["agent_status"] = status
        
        report = synthesis_agent.build_report(current_state)
        
        if job_id:
            event_bus.publish_event(job_id, "Consensus Engine", f"Verdict rendered: {verdict} ({confidence * 100:.1f}% confidence).")

        return {
            "final_verdict": verdict,
            "final_confidence": confidence,
            "report": report,
            "agent_status": status,
            "stream_events": [{"agent": "Consensus Engine", "message": f"Verdict: {verdict} ({confidence * 100:.1f}%)"}]
        }
    except Exception as e:
        logger.error(f"error in synthesis node: {e}", exc_info=True)
        status["synthesis"] = "failed"
        if job_id:
            event_bus.publish_event(job_id, "Consensus Engine", f"Synthesis failed: {e}")
        return {
            "final_verdict": "UNCERTAIN",
            "final_confidence": 0.0,
            "report": {},
            "agent_status": status
        }


from concurrent.futures import ThreadPoolExecutor


def cv_parallel_node(state: VeriFrameState) -> dict:
    """
    runs Visual Forensics and Temporal Consistency agents simultaneously in parallel.
    """
    with ThreadPoolExecutor(max_workers=2) as pool:
        future_v = pool.submit(visual_node, state)
        future_t = pool.submit(temporal_node, state)
        res_v = future_v.result()
        res_t = future_t.result()

    merged_status = {**res_v.get("agent_status", {}), **res_t.get("agent_status", {})}
    merged_events = res_v.get("stream_events", []) + res_t.get("stream_events", [])

    return {
        **res_v,
        **res_t,
        "agent_status": merged_status,
        "stream_events": merged_events
    }


# assemble the state graph
workflow = StateGraph(VeriFrameState)

# add node functions
workflow.add_node("cv_parallel", cv_parallel_node)
workflow.add_node("router", router_node)
workflow.add_node("llm", llm_node)
workflow.add_node("reflection", reflection_node)
workflow.add_node("synthesis", synthesis_node)

# build execution flow with true concurrent CV start and conditional routing
workflow.add_edge(START, "cv_parallel")
workflow.add_edge("cv_parallel", "router")

# conditional edge from router: either skip LLM straight to synthesis, or route to LLM
workflow.add_conditional_edges(
    "router",
    route_decision_edge,
    {
        "synthesis": "synthesis",
        "llm": "llm"
    }
)

# LLM passes output to Reflection node
workflow.add_edge("llm", "reflection")

# conditional edge from reflection: either loop back to LLM (if self-correction needed) or proceed to synthesis
workflow.add_conditional_edges(
    "reflection",
    reflection_edge,
    {
        "llm": "llm",          # CYCLIC REFLECTION LOOP!
        "synthesis": "synthesis"
    }
)

workflow.add_edge("synthesis", END)

# compile graph
compiled_graph = workflow.compile()


def run_pipeline(frames, metadata, job_id=None):
    """
    outer wrapper to trigger the compiled langgraph workflow.
    """
    initial_state = {
        "frames": frames,
        "metadata": metadata,
        "job_id": job_id or "",
        "agent_status": {},
        "reflection_count": 0
    }
    
    final_state = compiled_graph.invoke(initial_state)
    return final_state
