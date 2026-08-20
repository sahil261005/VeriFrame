# just a typed dict for the shared state between agents
# langgraph will use this to pass data around between nodes

from typing import TypedDict, List, Dict, Any, Optional, Annotated


def reduce_dict(left: dict, right: dict) -> dict:
    """
    helper to merge dict updates in parallel nodes.
    langgraph uses this as a reducer to avoid write conflicts.
    """
    if left is None:
        left = {}
    if right is None:
        right = {}
    
    # merge them together
    merged = dict(left)
    merged.update(right)
    return merged


def reduce_events(left: list, right: list) -> list:
    """
    reducer for stream events. appends new events to the existing list
    so parallel nodes can both push events without overwriting each other.
    """
    if left is None:
        left = []
    if right is None:
        right = []
    return left + right


class VeriFrameState(TypedDict, total=False):
    # input stuff
    video_path: str
    metadata: Dict[str, Any]
    frames: List[Dict[str, Any]]

    # visual agent output
    visual_score: float
    visual_flagged_frames: List[Dict[str, Any]]
    visual_per_frame: List[Dict[str, Any]]

    # temporal agent output
    temporal_score: float
    temporal_flagged_timestamps: List[float]
    flow_results: List[Dict[str, Any]]
    face_results: List[Dict[str, Any]]

    # audio agent output
    audio_score: float
    audio_details: Dict[str, Any]
    has_audio: bool

    # routing decisions (set by router node)
    route_decision: str       # "skip_llm", "llm_extended", or "llm_normal"
    llm_frame_count: int      # how many frames to send to LLM (8 or 12)

    # llm agent output
    llm_reasoning: str
    frame_explanations: Dict[str, str]
    llm_score: float
    tools_used: List[str]     # which tools the LLM chose to call

    # reflection loop tracking
    reflection_count: int           # how many times reflection has run (max 2)
    reflection_feedback: str        # corrective prompt from reflection node
    needs_correction: bool          # whether reflection found issues

    # synthesis output
    final_verdict: str
    final_confidence: float
    report: Dict[str, Any]

    # tracks which agents ran ok and which broke.
    # annotated with reduce_dict so parallel nodes can write to it at the same time.
    agent_status: Annotated[Dict[str, str], reduce_dict]

    # SSE streaming events for the frontend
    # annotated with reduce_events so parallel nodes can both push events
    stream_events: Annotated[List[Dict[str, Any]], reduce_events]
