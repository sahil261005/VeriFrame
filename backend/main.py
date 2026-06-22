import os
import shutil
import base64
import json
from datetime import datetime
from typing import List

from fastapi import FastAPI, Depends, HTTPException, status, File, UploadFile, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session

import config
import database
import models
import schemas
import auth
import preprocessing
import report
from agents.orchestrator import run_pipeline
import cv2

# initialize database tables on startup
database.create_tables()

app = FastAPI(title="VeriFrame API", description="Multi-agent deepfake detection platform")

# CORS middleware configuration for local testing
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/auth/register", response_model=schemas.TokenResponse)
def register(req: schemas.RegisterRequest, db: Session = Depends(database.get_db)):
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
    
    # generate JWT token
    access_token = auth.create_access_token(data={"sub": user.email})
    return {"access_token": access_token, "token_type": "bearer"}


@app.post("/auth/login", response_model=schemas.TokenResponse)
def login(req: schemas.LoginRequest, db: Session = Depends(database.get_db)):
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


# Swagger UI token URL endpoint
@app.post("/auth/swagger-token", response_model=schemas.TokenResponse)
def swagger_token(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(database.get_db)):
    """Swagger-compatible OAuth2 password token flow"""
    user = db.query(models.User).filter(models.User.email == form_data.username).first()
    if not user or not auth.verify_password(form_data.password, user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    access_token = auth.create_access_token(data={"sub": user.email})
    return {"access_token": access_token, "token_type": "bearer"}


# worker function to run the deepfake analysis in the background
def process_video_task(job_id: str, temp_path: str, meta: dict):
    db = database.SessionLocal()
    downscaled_path = None
    try:
        print(f"background processing started for job {job_id}...")
        
        # 1. downscale video to 480p to keep processing efficient
        downscaled_path = os.path.join(config.UPLOAD_DIR, f"downscaled_{job_id}.mp4")
        preprocessing.downscale_video(temp_path, downscaled_path)
        print(f"video downscaled for job {job_id}.")
        
        # 2. extract frames every 0.5 seconds
        # we extract at 0.5s intervals so the agents have enough resolution to track inconsistencies
        frames = preprocessing.extract_frames(downscaled_path, interval=0.5)
        print(f"extracted {len(frames)} frames for job {job_id}.")
        
        # 3. run langgraph multi-agent pipeline
        pipeline_output = run_pipeline(frames, meta)
        print(f"langgraph pipeline completed for job {job_id}.")
        
        # 4. load job to save outputs
        job = db.query(models.AnalysisJob).filter(models.AnalysisJob.id == job_id).first()
        if not job:
            print(f"error: job {job_id} not found in database.")
            return

        # 5. extract and build thumbnails for flagged suspicious frames
        # we fetch the raw frames from our extracted list matching the suspicious timestamps
        # which were chosen by visual and temporal agents (max 8)
        suspicious_frames = []
        visual_flagged = pipeline_output.get("visual_flagged_frames", [])
        temporal_flagged = pipeline_output.get("temporal_flagged_timestamps", [])
        
        from agents.llm_agent import pick_suspicious_frames
        flagged_frames = pick_suspicious_frames(visual_flagged, temporal_flagged, frames)
        
        thumbnails_list = []
        for f in flagged_frames:
            ts = round(f["timestamp"], 3)
            img_array = f["image"]
            
            # downscale image to 200px width for storing in DB
            h, w = img_array.shape[:2]
            target_w = 200
            target_h = int(h * (target_w / w))
            resized = cv2.resize(img_array, (target_w, target_h))
            
            # convert to jpeg with lower quality
            _, buffer = cv2.imencode(".jpg", resized, [cv2.IMWRITE_JPEG_QUALITY, 40])
            img_base64 = base64.b64encode(buffer).decode("utf-8")
            
            thumbnails_list.append({
                "timestamp": ts,
                "image_b64": f"data:image/jpeg;base64,{img_base64}"
            })
            
        # 6. update job metrics
        job.status = "completed"
        job.completed_at = datetime.utcnow()
        job.final_verdict = pipeline_output.get("final_verdict", "UNCERTAIN")
        job.confidence = pipeline_output.get("final_confidence", 0.0)
        job.is_partial_analysis = any(status == "failed" for status in pipeline_output.get("agent_status", {}).values())
        
        # serialise JSON columns to text
        job.report_json = json.dumps(pipeline_output.get("report", {}))
        job.flagged_frame_thumbnails = json.dumps(thumbnails_list)
        
        db.commit()
        print(f"job {job_id} saved successfully.")

    except Exception as e:
        print(f"error processing video for job {job_id}: {e}")
        # update job status to failed
        try:
            job = db.query(models.AnalysisJob).filter(models.AnalysisJob.id == job_id).first()
            if job:
                job.status = "failed"
                job.completed_at = datetime.utcnow()
                db.commit()
        except Exception as db_err:
            print(f"failed to update job error status: {db_err}")
            
    finally:
        db.close()
        # cleanup temporary files on disk
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception as e:
                print(f"error removing temp video {temp_path}: {e}")
                
        if downscaled_path and os.path.exists(downscaled_path):
            try:
                os.remove(downscaled_path)
            except Exception as e:
                print(f"error removing downscaled video {downscaled_path}: {e}")


@app.post("/upload", response_model=schemas.JobStatusResponse)
def upload_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    db: Session = Depends(database.get_db),
    current_user: models.User = Depends(auth.get_current_user)
):
    """upload video file to queue for deepfake analysis"""
    
    # 1. validate file extension format
    ext = os.path.splitext(file.filename)[1].lower().replace(".", "")
    allowed = ["mp4", "avi", "mov", "webm"]
    if ext not in allowed:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported video format: .{ext}. Only mp4, avi, mov, and webm are allowed."
        )
        
    # 2. save the uploaded file temporarily to run opencv validation
    import uuid
    job_uuid = str(uuid.uuid4())
    temp_filename = f"{job_uuid}_{file.filename}"
    temp_filepath = os.path.join(config.UPLOAD_DIR, temp_filename)
    
    try:
        with open(temp_filepath, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # 3. validate duration and structure
        meta = preprocessing.validate_video(temp_filepath)
        
    except ValueError as val_err:
        # validation failed, delete file and raise bad request
        if os.path.exists(temp_filepath):
            os.remove(temp_filepath)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(val_err)
        )
    except Exception as exc:
        if os.path.exists(temp_filepath):
            os.remove(temp_filepath)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error saving or reading upload file: {exc}"
        )
        
    # 4. create the database AnalysisJob row
    job = models.AnalysisJob(
        id=job_uuid,
        user_id=current_user.id,
        status="processing",
        video_filename=file.filename,
        duration=meta.get("duration", 0.0)
    )
    db.add(job)
    db.commit()
    db.refresh(job)
    
    # 5. dispatch background task
    background_tasks.add_task(process_video_task, job.id, temp_filepath, meta)
    
    return job


@app.get("/analysis/{job_id}", response_model=schemas.FullReportResponse)
def get_analysis(
    job_id: str,
    db: Session = Depends(database.get_db),
    current_user: models.User = Depends(auth.get_current_user)
):
    """retrieve analysis report by job ID"""
    job = db.query(models.AnalysisJob).filter(models.AnalysisJob.id == job_id).first()
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Analysis job not found"
        )
        
    # verify ownership
    if job.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to view this analysis report"
        )
        
    # parse the text JSON database fields back to dictionaries
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
    """export analysis report as a PDF file"""
    job = db.query(models.AnalysisJob).filter(models.AnalysisJob.id == job_id).first()
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Analysis job not found"
        )
        
    # verify ownership
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
        
    # unpack report
    report_dict = json.loads(job.report_json) if job.report_json else {}
    
    # generate PDF bytes
    pdf_data = report.generate_pdf(report_dict)
    
    # check if weasyprint returned fallback html or actual pdf
    # weasyprint returns PDF starting with %PDF
    is_pdf = pdf_data.startswith(b"%PDF")
    media_type = "application/pdf" if is_pdf else "text/html"
    filename = f"veriframe_report_{job_id}.pdf" if is_pdf else f"veriframe_report_{job_id}.html"
    
    return Response(
        content=pdf_data,
        media_type=media_type,
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )
