import threading
import time
import logging

logger = logging.getLogger(__name__)

# thread-safe in-memory event store
# each job has its own list of events that agents push to during pipeline execution
# the SSE endpoint reads from here to stream events to the frontend

_lock = threading.Lock()
_events = {}   # job_id -> list of event dicts
_status = {}   # job_id -> "running" | "completed" | "failed"


def init_job(job_id):
    """initialize event storage for a new job"""
    with _lock:
        _events[job_id] = []
        _status[job_id] = "running"


def publish_event(job_id, agent_name, message):
    """
    publish a progress event from an agent node.
    called by each agent node during execution to report what its doing.
    """
    event = {
        "agent": agent_name,
        "message": message,
        "timestamp": time.time()
    }
    with _lock:
        if job_id in _events:
            _events[job_id].append(event)
            logger.info(f"[SSE] [{agent_name}] {message}")


def mark_completed(job_id):
    """mark the job as done so the SSE endpoint knows to stop streaming"""
    with _lock:
        _status[job_id] = "completed"


def mark_failed(job_id):
    """mark the job as failed"""
    with _lock:
        _status[job_id] = "failed"


def get_events(job_id, after_index=0):
    """
    get events for a job starting from after_index.
    the SSE endpoint calls this in a loop and only sends new events to the client.
    returns (events_list, is_done)
    """
    with _lock:
        events = _events.get(job_id, [])
        new_events = events[after_index:]
        is_done = _status.get(job_id) in ("completed", "failed")
        final_status = _status.get(job_id, "running")
    return new_events, is_done, final_status


def cleanup_job(job_id):
    """remove event data for a job after its been fully consumed"""
    with _lock:
        _events.pop(job_id, None)
        _status.pop(job_id, None)
