from langgraph.graph import StateGraph, START, END
from agents.state import VeriFrameState
import agents.visual_agent as visual_agent
import agents.temporal_agent as temporal_agent
import agents.llm_agent as llm_agent
import agents.synthesis_agent as synthesis_agent


def visual_node(state: VeriFrameState) -> dict:
    """
    node for running visual analysis. checks deepfake classification + noise.
    """
    frames = state.get("frames", [])
    status = dict(state.get("agent_status", {}))
    
    try:
        pipe = visual_agent.load_model()
        score, flagged, all_results = visual_agent.analyze_frames(frames, pipe)
        status["visual"] = "success"
        
        return {
            "visual_score": score,
            "visual_flagged_frames": flagged,
            "visual_per_frame": all_results,
            "agent_status": status
        }
    except Exception as e:
        print(f"error in visual node: {e}")
        status["visual"] = "failed"
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
    
    try:
        score, flagged_times, flow_res, face_res = temporal_agent.run_temporal_analysis(frames)
        status["temporal"] = "success"
        
        return {
            "temporal_score": score,
            "temporal_flagged_timestamps": flagged_times,
            "flow_results": flow_res,
            "face_results": face_res,
            "agent_status": status
        }
    except Exception as e:
        print(f"error in temporal node: {e}")
        status["temporal"] = "failed"
        return {
            "temporal_score": 0.0,
            "temporal_flagged_timestamps": [],
            "flow_results": [],
            "face_results": [],
            "agent_status": status
        }


def llm_node(state: VeriFrameState) -> dict:
    """
    node for running gemini vision model on suspicious frames.
    """
    frames = state.get("frames", [])
    visual_flagged = state.get("visual_flagged_frames", [])
    temporal_flagged = state.get("temporal_flagged_timestamps", [])
    status = dict(state.get("agent_status", {}))

    try:
        # merge and pick the most suspicious frames (max 8)
        suspicious = llm_agent.pick_suspicious_frames(visual_flagged, temporal_flagged, frames)
        
        # analyze using gemini 2.5 flash
        reasoning, explanations, score = llm_agent.analyze_with_gemini(suspicious)
        status["llm"] = "success"
        
        return {
            "llm_score": score,
            "llm_reasoning": reasoning,
            "frame_explanations": explanations,
            "agent_status": status
        }
    except Exception as e:
        print(f"error in llm node: {e}")
        status["llm"] = "failed"
        return {
            "llm_score": 0.0,
            "llm_reasoning": f"gemini analysis failed: {e}",
            "frame_explanations": {},
            "agent_status": status
        }


def synthesis_node(state: VeriFrameState) -> dict:
    """
    node for synthesis. computes the final consensus verdict and compiles the report.
    """
    visual_score = state.get("visual_score", 0.0)
    temporal_score = state.get("temporal_score", 0.0)
    llm_score = state.get("llm_score", 0.0)
    status = dict(state.get("agent_status", {}))
    
    try:
        # compute weighted score and final verdict
        verdict, confidence, normalized_weights = synthesis_agent.compute_verdict(
            visual_score, temporal_score, llm_score, status
        )
        status["synthesis"] = "success"
        
        # update current state copy to build report cleanly
        current_state = dict(state)
        current_state["final_verdict"] = verdict
        current_state["final_confidence"] = confidence
        current_state["agent_status"] = status
        
        report = synthesis_agent.build_report(current_state)
        
        return {
            "final_verdict": verdict,
            "final_confidence": confidence,
            "report": report,
            "agent_status": status
        }
    except Exception as e:
        print(f"error in synthesis node: {e}")
        status["synthesis"] = "failed"
        return {
            "final_verdict": "UNCERTAIN",
            "final_confidence": 0.0,
            "report": {},
            "agent_status": status
        }


# assemble the state graph
workflow = StateGraph(VeriFrameState)

# add node functions
workflow.add_node("visual", visual_node)
workflow.add_node("temporal", temporal_node)
workflow.add_node("llm", llm_node)
workflow.add_node("synthesis", synthesis_node)

# build the execution flow (DAG)
# parallel start for visual and temporal checks
workflow.add_edge(START, "visual")
workflow.add_edge(START, "temporal")

# visual and temporal join together into the LLM node
workflow.add_edge("visual", "llm")
workflow.add_edge("temporal", "llm")

# sequential synthesis
workflow.add_edge("llm", "synthesis")
workflow.add_edge("synthesis", END)

# compile graph
compiled_graph = workflow.compile()


def run_pipeline(frames, metadata):
    """
    outer wrapper to trigger the compiled langgraph workflow.
    """
    initial_state = {
        "frames": frames,
        "metadata": metadata,
        "agent_status": {}
    }
    
    # run langgraph flow
    final_state = compiled_graph.invoke(initial_state)
    return final_state

