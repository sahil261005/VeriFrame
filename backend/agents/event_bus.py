import threading
import time
import logging

logger = logging.getLogger(__name__)

# simple in memory event store with a lock since agents run in different threads
# each job gets its own list of events that the agents push to while the pipeline runs
# the SSE endpoint reads from here and streams them to the frontend

_lock = threading.Lock()
_events = {}   # job_id -> list of event dicts
_status = {}   # job_id -> "running" | "completed" | "failed"
_finished_at = {}  # job_id -> time.time() when the job reached a terminal state

# keep finished jobs around for an hour so if the SSE client reconnects late it still gets the final event
_RETENTION_SECONDS = 3600


def _prune_finished_locked():
    cutoff = time.time() - _RETENTION_SECONDS
    for job_id in [j for j, t in _finished_at.items() if t < cutoff]:
        _events.pop(job_id, None)
        _status.pop(job_id, None)
        _finished_at.pop(job_id, None)


def init_job(job_id):
    """set up the event list for a new job"""
    with _lock:
        _prune_finished_locked()
        _events[job_id] = []
        _status[job_id] = "running"


def has_job(job_id):
    with _lock:
        return job_id in _status


def publish_event(job_id, agent_name, message):
    """
    each agent calls this while running to say what its doing.
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
        _finished_at[job_id] = time.time()


def mark_failed(job_id):
    """mark the job as failed"""
    with _lock:
        _status[job_id] = "failed"
        _finished_at[job_id] = time.time()


def get_events(job_id, after_index=0):
    """
    get events for a job starting from after_index.
    the SSE endpoint calls this in a loop so it only sends the new ones.
    returns (events, is_done, status)
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
        _finished_at.pop(job_id, None)
