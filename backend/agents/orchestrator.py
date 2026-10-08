from langgraph.graph import StateGraph, START, END
from agents.state import VeriFrameState
import agents.visual_agent as visual_agent
import agents.temporal_agent as temporal_agent
import agents.audio_agent as audio_agent
import agents.generative_agent as generative_agent
import agents.llm_agent as llm_agent
import agents.synthesis_agent as synthesis_agent
import agents.reflection_agent as reflection_agent
import agents.event_bus as event_bus
import logging
import time

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
        if pipe == "heuristic_fallback" or visual_agent.used_heuristics_only(all_results):
            status["visual"] = "fallback"
        elif visual_agent.no_faces_found(all_results):
            # face-swap classifier has nothing to judge; drop it from the consensus instead of voting "real"
            status["visual"] = "skipped"
        else:
            status["visual"] = "success"

        if job_id:
            if status["visual"] == "skipped":
                event_bus.publish_event(job_id, "Visual Forensics Agent", "Skipped: no face detected, face-swap classifier not applicable.")
            else:
                mode_desc = "ONNX ViT deep learning model" if status["visual"] != "fallback" else "spatial noise heuristics"
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
        # too few consecutive frames to measure anything (e.g. very short clips): don't vote "consistent"
        status["temporal"] = "success" if (flow_res or face_res) else "skipped"
        
        if job_id:
            if status["temporal"] == "skipped":
                event_bus.publish_event(job_id, "Temporal Consistency Agent", "Skipped: not enough consecutive frames to measure motion.")
            else:
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


def audio_node(state: VeriFrameState) -> dict:
    """
    node for running audio analysis: spectral frequency cutoffs + lip-sync correlation.
    """
    video_path = state.get("video_path", "")
    frames = state.get("frames", [])
    status = dict(state.get("agent_status", {}))
    job_id = state.get("job_id", "")

    logger.info(f"Audio Forensics Node started (video_path={video_path})...")
    if job_id:
        event_bus.publish_event(job_id, "Audio Forensics Agent", "Analyzing voice acoustics, spectral cutoffs & lip-sync...")

    try:
        score, details, has_audio = audio_agent.run_audio_analysis(video_path, frames)
        status["audio"] = "success" if has_audio else "skipped"
        
        summary_msg = details.get("summary", "Audio analysis completed.")
        logger.info(f"Audio Forensics Node result: has_audio={has_audio}, score={score:.2f}")
        if job_id:
            if has_audio:
                event_bus.publish_event(job_id, "Audio Forensics Agent", f"Completed. Audio fake score: {score:.2f}. {summary_msg}")
            else:
                event_bus.publish_event(job_id, "Audio Forensics Agent", "No audio track present (silent clip).")

        return {
            "audio_score": score,
            "audio_details": details,
            "has_audio": has_audio,
            "agent_status": status,
            "stream_events": [{"agent": "Audio Forensics", "message": f"Completed. Score: {score:.2f}" if has_audio else "Silent clip"}]
        }
    except Exception as e:
        logger.error(f"error in audio node: {e}", exc_info=True)
        status["audio"] = "failed"
        if job_id:
            event_bus.publish_event(job_id, "Audio Forensics Agent", f"Failed: {e}")
        return {
            "audio_score": 0.0,
            "audio_details": {},
            "has_audio": False,
            "agent_status": status
        }


def generative_node(state: VeriFrameState) -> dict:
    """
    node for the generative-video detector: scores whole frames for AI-generation artifacts
    (the face-swap ViT in visual_node cannot see Sora/Kling/Runway-style footage).
    """
    frames = state.get("frames", [])
    status = dict(state.get("agent_status", {}))
    job_id = state.get("job_id", "")

    if job_id:
        event_bus.publish_event(job_id, "Generative Video Agent", "Scanning frames for diffusion / AI video-model artifacts...")

    empty = {"generative_score": 0.0, "generative_flagged_frames": [], "generative_per_frame": []}
    try:
        session = generative_agent.load_model()
        if session is None:
            status["generative"] = "skipped"
            if job_id:
                event_bus.publish_event(job_id, "Generative Video Agent", "Skipped: generative detector model not installed.")
            return {**empty, "agent_status": status}

        score, per_frame = generative_agent.analyze_frames(frames, session)
        status["generative"] = "success"
        scored = [r for r in per_frame if r["generated_probability"] is not None]
        flagged = [
            {"timestamp": r["timestamp"], "fake_confidence": r["generated_probability"]}
            for r in sorted(scored, key=lambda r: r["generated_probability"], reverse=True)[:5]
        ]
        if job_id:
            event_bus.publish_event(job_id, "Generative Video Agent", f"Completed. AI-generation score: {score:.2f}")
        return {
            "generative_score": score,
            "generative_flagged_frames": flagged,
            "generative_per_frame": per_frame,
            "agent_status": status,
            "stream_events": [{"agent": "Generative Video", "message": f"Completed. Score: {score:.2f}"}]
        }
    except Exception as e:
        logger.error(f"error in generative node: {e}", exc_info=True)
        status["generative"] = "failed"
        if job_id:
            event_bus.publish_event(job_id, "Generative Video Agent", f"Failed: {e}")
        return {**empty, "agent_status": status}


def router_node(state: VeriFrameState) -> dict:
    """
    conditional router node. inspects visual and temporal scores and decides
    how many frames to send to the LLM agent. always routes to LLM for
    multi-agent consensus - never skips it.
    """
    visual_score = state.get("visual_score", 0.0)
    temporal_score = state.get("temporal_score", 0.0)
    status = state.get("agent_status", {})
    job_id = state.get("job_id", "")

    logger.info(f"Router check: visual_score={visual_score}, temporal_score={temporal_score}")

    # if the visual model did not produce a real verdict (fallback/skipped/failed), use 4 frames for safety
    if status.get("visual") != "success":
        decision = "llm_extended"
        frame_count = 4
        reason = f"Visual agent status '{status.get('visual', 'unknown')}'. Evaluating 4 keyframes with LLM."
    # if visual and temporal both have clear high confidence (>0.85) or both clear low (<0.15)
    elif (visual_score > 0.85 and temporal_score > 0.60) or (visual_score < 0.15 and temporal_score < 0.15):
        # minimum 3 frames: the face-swap model cannot see fully AI-generated video, so its "clean"
        # verdict is not a reason to give the LLM (the main AI-video detector) less evidence
        decision = "llm_fast_consensus"
        frame_count = 3
        reason = "Strong CV agreement detected. Running 3-keyframe LLM verification."
    else:
        decision = "llm_normal"
        frame_count = 3
        reason = "Routing to LLM ReAct Vision agent with 3 keyframe samples."

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
    # frames either image detector found most suspicious go to the LLM first
    visual_flagged = sorted(
        state.get("visual_flagged_frames", []) + _generative_result(state, wait=False).get("generative_flagged_frames", []),
        key=lambda f: f.get("fake_confidence", 0.0), reverse=True
    )
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

    audio_details = state.get("audio_details", {})
    has_audio = state.get("has_audio", False)

    # token counts of every API call made in this run (kept even if the node fails afterwards)
    usage = {}
    try:
        suspicious = llm_agent.pick_suspicious_frames(visual_flagged, temporal_flagged, frames, max_count=max_count)
        
        reasoning, explanations, score, tools_used = llm_agent.analyze_with_llm(
            suspicious, 
            reflection_prompt=reflection_feedback, 
            metadata=metadata, 
            all_frames=frames,
            audio_details=audio_details if has_audio else None,
            usage=usage
        )
        status["llm"] = "success"

        if job_id:
            event_bus.publish_event(job_id, "Cognitive Reasoning Agent", f"Completed tool-calling analysis. Score: {score:.2f}. Tools invoked: {', '.join(tools_used)}")
        
        return {
            "llm_score": score,
            "llm_reasoning": reasoning,
            "frame_explanations": explanations,
            "tools_used": tools_used,
            "llm_usage": usage,
            "agent_status": status,
            "stream_events": [{"agent": "Cognitive Reasoning", "message": f"Completed using ReAct tools. Score: {score:.2f}"}]
        }
    except llm_agent.LLMUnavailableError as e:
        logger.warning(str(e))
        status["llm"] = "skipped"
        if job_id:
            event_bus.publish_event(job_id, "Cognitive Reasoning Agent", "Skipped: no LLM API key configured.")
        return {
            "llm_score": 0.0,
            "llm_reasoning": str(e),
            "frame_explanations": {},
            "tools_used": [],
            "agent_status": status
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
            "llm_usage": usage,
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
    audio_score = state.get("audio_score", 0.0)
    has_audio = state.get("has_audio", False)
    llm_score = state.get("llm_score", 0.0)
    # collect the background generative detector (usually already finished while the LLM was running)
    generative = _generative_result(state, wait=True)
    generative_score = generative.get("generative_score", 0.0)
    status = {**state.get("agent_status", {}), **generative.get("agent_status", {})}
    job_id = state.get("job_id", "")

    if job_id:
        event_bus.publish_event(job_id, "Consensus Engine", "Synthesizing multi-agent weights and building final verdict report...")

    try:
        metadata = state.get("metadata", {})
        verdict, confidence, normalized_weights = synthesis_agent.compute_verdict(
            visual_score, temporal_score, llm_score, status, metadata,
            audio_score=audio_score, has_audio=has_audio,
            generative_score=generative_score
        )
        status["synthesis"] = "success"
        
        current_state = {**state, **generative}
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
            "generative_score": generative_score,
            "generative_flagged_frames": generative.get("generative_flagged_frames", []),
            "generative_per_frame": generative.get("generative_per_frame", []),
            "timings": generative.get("timings", {}),
            "agent_status": status,
            "stream_events": generative.get("stream_events", []) + [{"agent": "Consensus Engine", "message": f"Verdict: {verdict} ({confidence * 100:.1f}%)"}]
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


# the AI-generated-video detector runs on the hosted service and is the slowest CV step. it does not need to
# block the LLM stage (~4-5s), so it runs in the background and the synthesis node collects it at the end.
_background = ThreadPoolExecutor(max_workers=2)


def _generative_result(state, wait):
    """result dict of the background generative node; with wait=False returns {} if it is not finished yet."""
    future = state.get("generative_future")
    if future is None or (not wait and not future.done()):
        return {}
    return future.result()


def cv_parallel_node(state: VeriFrameState) -> dict:
    """
    runs Visual Forensics, Temporal Consistency, and Audio Forensics simultaneously in parallel.
    audio takes ~0.4s and finishes while visual is running, adding 0s to overall wait.
    the generative-video detector is started alongside them but collected later (see synthesis_node).
    """
    generative_future = _background.submit(_timed("generative", generative_node), state)

    with ThreadPoolExecutor(max_workers=3) as pool:
        future_v = pool.submit(visual_node, state)
        future_t = pool.submit(temporal_node, state)
        future_a = pool.submit(audio_node, state)
        
        res_v = future_v.result()
        res_t = future_t.result()
        res_a = future_a.result()

    merged_status = {
        **res_v.get("agent_status", {}),
        **res_t.get("agent_status", {}),
        **res_a.get("agent_status", {})
    }
    merged_events = res_v.get("stream_events", []) + res_t.get("stream_events", []) + res_a.get("stream_events", [])

    return {
        **res_v,
        **res_t,
        **res_a,
        "generative_future": generative_future,
        "agent_status": merged_status,
        "stream_events": merged_events
    }


def _timed(name, node_fn):
    """wraps a node so its wall time is added to state["timings"][name] (summed if the node re-runs)."""
    def wrapper(state):
        start = time.perf_counter()
        result = node_fn(state)
        timings = dict(result.get("timings", {}))
        timings[name] = timings.get(name, 0.0) + round(time.perf_counter() - start, 3)
        return {**result, "timings": timings}
    return wrapper


# assemble the state graph
workflow = StateGraph(VeriFrameState)

# add node functions
workflow.add_node("cv_parallel", _timed("cv_parallel", cv_parallel_node))
workflow.add_node("router", _timed("router", router_node))
workflow.add_node("llm", _timed("llm", llm_node))
workflow.add_node("reflection", _timed("reflection", reflection_node))
workflow.add_node("synthesis", _timed("synthesis", synthesis_node))

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


def run_pipeline(frames, metadata, job_id=None, video_path=""):
    """
    outer wrapper to trigger the compiled langgraph workflow.
    """
    initial_state = {
        "video_path": video_path or "",
        "frames": frames,
        "metadata": metadata,
        "job_id": job_id or "",
        "agent_status": {},
        "reflection_count": 0
    }
    
    start = time.perf_counter()
    final_state = compiled_graph.invoke(initial_state)

    # latency and LLM cost for this video, shown in the report and collected by evaluate_pipeline.py
    usage = final_state.get("llm_usage", {}) or {}
    performance = {
        "pipeline_seconds": round(time.perf_counter() - start, 3),
        "stage_seconds": final_state.get("timings", {}),
        "llm_calls": int(usage.get("llm_calls", 0)),
        "llm_input_tokens": int(sum(v for k, v in usage.items() if k.endswith("_input_tokens"))),
        "llm_output_tokens": int(sum(v for k, v in usage.items() if k.endswith("_output_tokens"))),
        "llm_cost_usd": llm_agent.estimate_cost_usd(usage),
    }
    final_state["performance"] = performance
    if isinstance(final_state.get("report"), dict):
        final_state["report"]["performance"] = performance
    return final_state
