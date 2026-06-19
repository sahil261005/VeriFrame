# just a typed dict for the shared state between agents
# langgraph will use this later in phase 3 to pass data around

from typing import TypedDict, List, Dict, Any, Optional


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

    # llm agent output (phase 3)
    llm_reasoning: str
    frame_explanations: Dict[str, str]

    # synthesis output (phase 3)
    final_verdict: str
    final_confidence: float
    report: Dict[str, Any]

    # tracks which agents ran ok and which broke
    agent_status: Dict[str, str]
