import os
import shutil
import base64
import json
from datetime import datetime
from typing import List
import logging
import time
from dotenv import load_dotenv

load_dotenv()

import config

# has to be set before numpy, opencv and onnxruntime get imported, see config.CPU_THREADS
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, str(config.CPU_THREADS))

# langsmith tracing is off for good. langgraph would copy and upload every nodes state with all the video frames
# on each step, which tripled memory use and sent user videos to a third party
for _var in ("LANGCHAIN_TRACING_V2", "LANGCHAIN_TRACING", "LANGSMITH_TRACING"):
    os.environ[_var] = "false"
for _var in ("LANGCHAIN_API_KEY", "LANGSMITH_API_KEY"):
    os.environ.pop(_var, None)

# basic logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

from fastapi import FastAPI, Depends, HTTPException, status, File, UploadFile, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse, JSONResponse
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy import text
from sqlalchemy.orm import Session
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded

import config
import database
import models
import schemas
import auth
import preprocessing
import report
import agents.event_bus as event_bus
from agents.orchestrator import run_pipeline
import cv2

# create the db tables on startup
database.create_tables()

def _client_ip(request: Request) -> str:
    """on render every request comes from their proxy so we use the ip it added to X-Forwarded-For"""
    forwarded = request.headers.get("x-forwarded-for", "")
    if os.environ.get("RENDER") and forwarded:
        return forwarded.split(",")[-1].strip()
    return request.client.host if request.client else "unknown"


limiter = Limiter(key_func=_client_ip)
app = FastAPI(title="VeriFrame API", description="Multi-agent deepfake detection platform")
app.state.limiter = limiter

@app.exception_handler(RateLimitExceeded)
def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    # raising in here turns into a 500 so we just return the 429 ourselves
    return JSONResponse(
        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
        content={"detail": "Too many requests. Please wait a minute and try again."}
    )


# cors setup for local testing
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


import threading
import agents.visual_agent as visual_agent
import agents.generative_agent as generative_agent
import agents.remote_detector as remote_detector


def _fail_orphaned_jobs():
    """jobs still "processing" at boot are from an old process that died, like when it ran out of memory.
    nothing will ever finish them so we mark them failed and the page shows an error instead of loading forever"""
    db = database.SessionLocal()
    try:
        orphans = db.query(models.AnalysisJob).filter(models.AnalysisJob.status == "processing").all()
        for job in orphans:
            job.status = "failed"
            job.completed_at = datetime.utcnow()
        db.commit()
        if orphans:
            logger.warning(f"marked {len(orphans)} interrupted job(s) as failed after restart")
    except Exception as e:
        logger.error(f"could not clean up interrupted jobs: {e}")
    finally:
        db.close()


def _clear_old_uploads():
    # if the server died mid job its temp video is still sitting in the upload folder
    # nothing is running yet at startup so anything in there is left over
    for name in os.listdir(config.UPLOAD_DIR):
        path = os.path.join(config.UPLOAD_DIR, name)
        try:
            if os.path.isfile(path):
                os.remove(path)
        except Exception as e:
            logger.error(f"could not remove old upload {name}: {e}")


@app.on_event("startup")
def startup_event():
    """load the onnx deepfake model into memory when the server boots"""
    _fail_orphaned_jobs()
    _clear_old_uploads()
    threading.Thread(target=visual_agent.load_model, daemon=True).start()
    # wake up the hosted generative video detector since free hugging face spaces sleep when idle
    threading.Thread(target=remote_detector.warm_up, daemon=True).start()


@app.get("/health")
@limiter.exempt
def health_check():
    """health check for the github actions keep-alive ping, also says if the onnx model is loaded.
    its exempt from rate limiting so the pings never get blocked"""
    model_loaded = visual_agent._onnx_session is not None

    # touch the database too. the keep-alive job hits this every 5 min, so supabase sees activity
    # and doesnt pause the free project when nobody has used the site for a while
    db_ok = True
    db = database.SessionLocal()
    try:
        db.execute(text("SELECT 1"))
    except Exception as e:
        db_ok = False
        logger.error(f"health check could not reach the database: {e}")
    finally:
        db.close()

    return {"status": "ok", "service": "VeriFrame API", "model_loaded": model_loaded, "database": db_ok}


@app.post("/auth/register", response_model=schemas.TokenResponse)
@limiter.limit("10/hour")
def register(request: Request, req: schemas.RegisterRequest, db: Session = Depends(database.get_db)):
    """register a new user"""
    # check if user already exists
    existing = db.query(models.User).filter(models.User.email == req.email).first()
    if existing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )
    
    # create new user
    hashed_pwd = auth.get_password_hash(req.password)
    user = models.User(email=req.email, password_hash=hashed_pwd)
    db.add(user)
    db.commit()
    db.refresh(user)
    
    # make the jwt token
    access_token = auth.create_access_token(data={"sub": user.email})
    return {"access_token": access_token, "token_type": "bearer"}


@app.post("/auth/login", response_model=schemas.TokenResponse)
@limiter.limit("10/minute")
def login(request: Request, req: schemas.LoginRequest, db: Session = Depends(database.get_db)):
    """login user and return token"""
    user = db.query(models.User).filter(models.User.email == req.email).first()
    if not user or not auth.verify_password(req.password, user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    access_token = auth.create_access_token(data={"sub": user.email})
    return {"access_token": access_token, "token_type": "bearer"}


# token route for the swagger ui login
@app.post("/auth/swagger-token", response_model=schemas.TokenResponse)
@limiter.limit("10/minute")
def swagger_token(request: Request, form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(database.get_db)):
    """same as login but takes the form data swagger sends"""
    user = db.query(models.User).filter(models.User.email == form_data.username).first()
    if not user or not auth.verify_password(form_data.password, user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    access_token = auth.create_access_token(data={"sub": user.email})
    return {"access_token": access_token, "token_type": "bearer"}


# a 512mb server cant hold a few decoded videos plus the cv models at once so we run one analysis at a time
# extra uploads just wait here as "processing" instead of crashing the server
_analysis_slot = threading.Semaphore(int(os.environ.get("MAX_CONCURRENT_ANALYSES", "1")))


# every queued analysis holds a worker thread while it waits, so unlimited uploads would freeze the whole server
MAX_PENDING_JOBS = int(os.environ.get("MAX_PENDING_JOBS", "6"))
MAX_UPLOAD_BYTES = 50 * 1024 * 1024
_pending_jobs = 0
_pending_lock = threading.Lock()


def process_video_task(job_id: str, temp_path: str, meta: dict):
    global _pending_jobs
    try:
        with _analysis_slot:
            _process_video(job_id, temp_path, meta)
    finally:
        with _pending_lock:
            _pending_jobs -= 1


def _process_video(job_id: str, temp_path: str, meta: dict):
    db = database.SessionLocal()
    try:
        logger.info(f"background processing started for job {job_id}...")
        
        # 1. grab some keyframes from the video and resize them in memory
        job_start = time.perf_counter()
        frames = preprocessing.extract_frames(temp_path, interval=1.0, target_height=480, max_frames=6)
        preprocessing.release_memory()  # give the decoder memory back before the models and llm run
        extract_seconds = round(time.perf_counter() - job_start, 3)
        logger.info(f"extracted {len(frames)} keyframes for job {job_id}.")
        if not frames:
            # otherwise a broken or empty video goes through all the agents with nothing to look at
            raise ValueError("could not read any frames from this video")
        
        # 2. run the langgraph agents, we pass the video path too for the audio check
        pipeline_output = run_pipeline(frames, meta, job_id=job_id, video_path=temp_path)
        logger.info(f"langgraph pipeline completed for job {job_id}.")
        
        # 3. load the job so we can save the results
        job = db.query(models.AnalysisJob).filter(models.AnalysisJob.id == job_id).first()
        if not job:
            logger.error(f"error: job {job_id} not found in database.")
            event_bus.mark_failed(job_id)
            return

        # 4. thumbnails of the frames the llm actually looked at so every thumbnail has a real explanation
        llm_ts = set(pipeline_output.get("llm_frame_timestamps") or [])
        if llm_ts:
            flagged_frames = [f for f in frames if round(f["timestamp"], 3) in llm_ts]
        else:
            # llm got skipped or failed so just show the frames it would have gotten
            from agents.llm_agent import pick_suspicious_frames
            flagged_frames = pick_suspicious_frames(
                pipeline_output.get("visual_flagged_frames", []),
                pipeline_output.get("temporal_flagged_timestamps", []),
                frames,
                max_count=pipeline_output.get("llm_frame_count", 4),
            )
        
        thumbnails_list = []
        for f in flagged_frames:
            ts = round(f["timestamp"], 3)
            img_array = f["image"]
            
            h, w = img_array.shape[:2]
            target_w = 480  # these show as cards on the results page
            target_h = int(h * (target_w / w))
            resized = cv2.resize(img_array, (target_w, target_h))
            
            _, buffer = cv2.imencode(".jpg", resized, [cv2.IMWRITE_JPEG_QUALITY, 70])
            img_base64 = base64.b64encode(buffer).decode("utf-8")
            
            thumbnails_list.append({
                "timestamp": ts,
                "image_b64": f"data:image/jpeg;base64,{img_base64}"
            })
            
        # 5. save the results on the job
        job.status = "completed"
        job.completed_at = datetime.utcnow()
        job.final_verdict = pipeline_output.get("final_verdict", "UNCERTAIN")
        job.confidence = pipeline_output.get("final_confidence", 0.0)
        job.is_partial_analysis = any(status == "failed" for status in pipeline_output.get("agent_status", {}).values())
        
        report = pipeline_output.get("report", {})
        if isinstance(report.get("performance"), dict):
            report["performance"]["extract_seconds"] = extract_seconds
            report["performance"]["total_seconds"] = round(time.perf_counter() - job_start, 3)
        job.report_json = json.dumps(report)
        job.flagged_frame_thumbnails = json.dumps(thumbnails_list)
        
        db.commit()
        logger.info(f"job {job_id} saved successfully.")
        event_bus.mark_completed(job_id)

    except Exception as e:
        logger.error(f"error processing video for job {job_id}: {e}", exc_info=True)
        event_bus.mark_failed(job_id)
        try:
            db.rollback()  # if the error came from a db commit the session is stuck until we roll back
            job = db.query(models.AnalysisJob).filter(models.AnalysisJob.id == job_id).first()
            if job:
                job.status = "failed"
                job.completed_at = datetime.utcnow()
                db.commit()
        except Exception as db_err:
            logger.error(f"failed to update job error status: {db_err}", exc_info=True)
            
    finally:
        db.close()
        # clean up the uploaded temp file
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception as e:
                logger.error(f"error removing temp video {temp_path}: {e}")
        preprocessing.release_memory()


@app.post("/upload", response_model=schemas.JobStatusResponse)
@limiter.limit("5/minute")
def upload_video(
    request: Request,
    file: UploadFile = File(...),
    db: Session = Depends(database.get_db),
    current_user: models.User = Depends(auth.get_current_user)
):
    """upload a video and queue it for analysis"""
    
    ext = os.path.splitext(file.filename)[1].lower().replace(".", "")
    allowed = ["mp4", "avi", "mov", "webm"]
    if ext not in allowed:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported video format: .{ext}. Only mp4, avi, mov, and webm are allowed."
        )
        
    global _pending_jobs
    with _pending_lock:
        if _pending_jobs >= MAX_PENDING_JOBS:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="VeriFrame is busy analysing other videos. Please try again in a minute."
            )
        _pending_jobs += 1

    import uuid
    job_uuid = str(uuid.uuid4())
    # we make the filename ourselves so a client filename like "../../x" cant choose where the file goes
    temp_filepath = os.path.join(config.UPLOAD_DIR, f"{job_uuid}.{ext}")

    try:
        written = 0
        with open(temp_filepath, "wb") as buffer:
            while chunk := file.file.read(1024 * 1024):
                written += len(chunk)
                if written > MAX_UPLOAD_BYTES:
                    raise ValueError(f"Video is too large. The limit is {MAX_UPLOAD_BYTES // (1024 * 1024)} MB.")
                buffer.write(chunk)

        meta = preprocessing.validate_video(temp_filepath)

    except ValueError as val_err:
        if os.path.exists(temp_filepath):
            os.remove(temp_filepath)
        with _pending_lock:
            _pending_jobs -= 1
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(val_err)
        )
    except Exception:
        logger.exception("error saving or reading upload for job %s", job_uuid)
        if os.path.exists(temp_filepath):
            os.remove(temp_filepath)
        with _pending_lock:
            _pending_jobs -= 1
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not process the uploaded video."
        )
        
    # set up the sse event bus for this job
    event_bus.init_job(job_uuid)

    try:
        job = models.AnalysisJob(
            id=job_uuid,
            user_id=current_user.id,
            status="processing",
            video_filename=os.path.basename((file.filename or "video").replace("\\", "/"))[-120:],
            duration=meta.get("duration", 0.0)
        )
        db.add(job)
        db.commit()
        db.refresh(job)
    except Exception:
        # if the job never saves the worker never starts so we have to give the queue slot back here
        logger.exception("could not save job %s", job_uuid)
        db.rollback()
        if os.path.exists(temp_filepath):
            os.remove(temp_filepath)
        with _pending_lock:
            _pending_jobs -= 1
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not start the analysis. Please try again."
        )

    # using our own thread instead of fastapi background tasks because those dont run if the browser
    # disconnects before the response is sent, and then the job would be stuck on "processing"
    worker = threading.Thread(target=process_video_task, args=(job.id, temp_filepath, meta), daemon=True)
    worker.start()

    return job


@app.get("/stream/{job_id}")
async def stream_job_events(job_id: str):
    """sse endpoint that streams the live agent events to the frontend"""
    import asyncio

    async def event_generator():
        last_index = 0
        while True:
            if not event_bus.has_job(job_id):
                # unknown job, like after a server restart. just close it instead of streaming forever and the client goes back to polling
                yield f"data: {json.dumps({'agent': 'System', 'message': 'Live event stream unavailable for this job.'})}\n\n"
                break

            new_events, is_done, status = event_bus.get_events(job_id, after_index=last_index)
            
            for event in new_events:
                last_index += 1
                yield f"data: {json.dumps(event)}\n\n"

            if is_done:
                yield f"data: {json.dumps({'agent': 'System', 'message': f'Job {status}', 'status': status})}\n\n"
                break

            await asyncio.sleep(0.8)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


@app.get("/analysis/{job_id}", response_model=schemas.FullReportResponse)
def get_analysis(
    job_id: str,
    db: Session = Depends(database.get_db),
    current_user: models.User = Depends(auth.get_current_user)
):
    """get the analysis report for a job"""
    job = db.query(models.AnalysisJob).filter(models.AnalysisJob.id == job_id).first()
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Analysis job not found"
        )
        
    # make sure this job belongs to the user
    if job.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to view this analysis report"
        )
        
    # the db stores these as json text so turn them back into dicts
    report_dict = json.loads(job.report_json) if job.report_json else None
    thumbnails = json.loads(job.flagged_frame_thumbnails) if job.flagged_frame_thumbnails else None
    
    return schemas.FullReportResponse(
        id=job.id,
        status=job.status,
        video_filename=job.video_filename,
        duration=job.duration,
        created_at=job.created_at,
        completed_at=job.completed_at,
        final_verdict=job.final_verdict,
        confidence=job.confidence,
        is_partial_analysis=job.is_partial_analysis,
        report=report_dict,
        thumbnails=thumbnails
    )


@app.get("/report/{job_id}/pdf")
def get_report_pdf(
    job_id: str,
    db: Session = Depends(database.get_db),
    current_user: models.User = Depends(auth.get_current_user)
):
    """download the report as a pdf"""
    job = db.query(models.AnalysisJob).filter(models.AnalysisJob.id == job_id).first()
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Analysis job not found"
        )
        
    # make sure this job belongs to the user
    if job.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to access this report"
        )
        
    if job.status != "completed":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Report cannot be generated while job is in status: {job.status}"
        )
        
    report_dict = json.loads(job.report_json) if job.report_json else {}
    
    pdf_data = report.generate_pdf(report_dict)
    
    # weasyprint might fail and give us html instead, a real pdf starts with %PDF
    is_pdf = pdf_data.startswith(b"%PDF")
    media_type = "application/pdf" if is_pdf else "text/html"
    filename = f"veriframe_report_{job_id}.pdf" if is_pdf else f"veriframe_report_{job_id}.html"
    
    return Response(
        content=pdf_data,
        media_type=media_type,
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )
