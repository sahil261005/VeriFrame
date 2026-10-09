# VeriFrame: Multi-Agent Video Forensics & Deepfake Detection Platform

VeriFrame is an explainable multi-agent system that decides whether a video is authentic, a face-swap deepfake, or fully AI-generated (Sora, Veo, Kling, Wan and similar). Instead of one black-box score, a team of specialist agents each examine a different kind of evidence. A LangGraph state graph coordinates them, and a consensus engine combines their findings into a verdict with per-frame explanations.

**Deployed architecture:** React frontend on Vercel, FastAPI backend on Render, Postgres on Supabase, with Gemini and Groq as the reasoning APIs. An optional AI-generation detector (client code in `generative_agent.py` / `remote_detector.py`, expecting an external scoring service) is **not part of the deployed configuration** (see the ablation below for why).

### In plain words
Think of it as a **panel of experts** looking at the same video:

1. You upload a video (up to 30 s and 1080p).
2. The app reads the video one frame at a time and keeps 6 key frames plus a few frames right after each one, shrunk to 480p as they are read. The video file itself is never re-encoded.
3. Several experts examine them **at the same time**:
   - one checks faces for swaps;
   - one checks the motion between frames for flicker;
   - one listens to the audio;
   - one reads the file's signed "made by AI" label (C2PA), if it has one.
4. The **3-4 most suspicious frames** go to an AI vision model. Gemini is tried first, and Groq takes over automatically if Gemini fails. The model explains what looks wrong in each frame.
5. A **reviewer** checks that the AI's explanation matches its score, and asks again if they contradict.
6. A **moderator** (the consensus engine) weighs everyone's opinion. It ignores any expert that couldn't do its job, then answers **AUTHENTIC**, **MANIPULATED** or **UNCERTAIN**, with a reason for each frame.

On a 50-video test set it was right 88% of the time and wrongly flagged only 1 of 25 real videos. A typical answer takes about 3.4 seconds on a laptop. On Render's free tier, which gives the app only a small share of a CPU, a 13-second clip takes about 47-50 seconds (see [Production Safeguards](#production-safeguards-for-a-512-mb-host)).

---

## Key Highlights

*   **Seven specialist components, one shared state.** Face-swap classifier, AI-generation detector, temporal consistency, audio forensics, provenance (C2PA), a vision-LLM reasoner and a self-correction (reflection) node, merged by a consensus engine.
*   **Explainable output.** Every verdict comes with per-frame, human-readable reasoning and a per-agent score breakdown, streamed live to the UI over Server-Sent Events.
*   **Graceful degradation.** If an agent fails or is skipped (no audio track, no face in the video, API down), its weight is redistributed to the others. If no agent produced evidence, the answer is `UNCERTAIN`, never a false "authentic".
*   **Measured, reproducible evaluation.** A 50-video benchmark built from public datasets, with a script that rebuilds the set and prints accuracy, per-generator detection rate, latency and token usage (see [Evaluation](#evaluation)).
*   **Built for free tiers.** Memory was profiled and tuned (duplicate model weights removed, tracing that copied every frame removed, one analysis at a time, upload caps). After the memory fixes, a 1080p job peaked at about 490 MB on a Mac (see Production Safeguards), which leaves little room under Render's 512 MB limit, so the real figure should be read from Render's Metrics tab. CPU use is capped to what the host actually gives the container (see [Production Safeguards](#production-safeguards-for-a-512-mb-host)).
*   **A report-style interface.** A light, paper-and-ink design. The score bars carry the real consensus cut-offs (0.30 and 0.48), the live screen shows each pipeline stage as it runs, and the report opens automatically when the analysis finishes.

---

## Technology Stack

### Backend
*   **API:** FastAPI, Uvicorn, background tasks, Server-Sent Events for live progress.
*   **Database:** SQLAlchemy. SQLite locally, Supabase Postgres in production (via `DATABASE_URL`).
*   **Orchestration:** LangGraph (shared typed state, parallel nodes, conditional routing, a cyclic reflection loop).
*   **Computer vision / ML:** OpenCV (frame extraction, optical flow), MediaPipe (face detection and landmarks), ONNX Runtime (INT8 Vision Transformer).
*   **LLMs:**
    *   **Primary: Google Gemini `gemini-3.5-flash-lite`.** It looks at all selected frames in one call and returns JSON.
    *   **Automatic fallback: Groq `qwen/qwen3.8-27b`.** It is a vision model, scores one frame per call, and runs the calls in parallel. Override it with `GROQ_VISION_MODEL`. Groq retired Llama 4 Scout, which used to be the default.
    *   Groq takes over whenever the Gemini call raises an error, such as a rate limit, a bad key or an outage.
    *   **Tested on 2026-10-08:**
        *   A forced Gemini failure (an invalid key) switched to Groq.
        *   Groq returned a scored explanation in about 1.6 s, using about 1,400 input tokens per frame.
*   **Provenance:** `c2pa-python` (reads signed C2PA Content Credentials).
*   **Reports and safety:** WeasyPrint (PDF, loaded lazily), `slowapi` rate limiting (login, register, upload), a cap on queued analyses, bcrypt and JWT authentication. Details in [Production Safeguards](#production-safeguards-for-a-512-mb-host).

### Frontend
*   React with Vite, Axios (JWT interceptors), EventSource for live progress, hand-written CSS with design tokens (Archivo and IBM Plex fonts, light theme).

### Hosting
| Part | Where | Notes |
|---|---|---|
| Frontend | Vercel (free) | Static site |
| API + orchestration + face-swap ViT | Render (free, 512 MB) | The 83 MB INT8 ONNX model runs here |
| Database | Supabase (free) | Hosted Postgres |
| LLM reasoning | Gemini API, Groq fallback | External APIs |
| Keep-alive | GitHub Actions (`.github/workflows/keepalive.yml`) | Pings `/health` every 5 minutes so Render's free instance does not go to sleep |

---

## Project Structure

```
backend/
├── agents/
│   ├── orchestrator.py       # LangGraph graph, nodes, routing, per-node timing
│   ├── state.py              # Shared typed state + reducers (merge dicts, sum timings)
│   ├── visual_agent.py       # Face-swap ViT (local ONNX or hosted) + face-presence gating
│   ├── generative_agent.py   # AI-generated-video detector (optional: external service or local ONNX)
│   ├── remote_detector.py    # HTTP client for the hosted detectors (PNG upload, retry, warm-up)
│   ├── temporal_agent.py     # Optical-flow spikes + facial-landmark jitter on consecutive frames
│   ├── audio_agent.py        # Spectral cutoff, voice cadence, lip-sync checks
│   ├── llm_agent.py          # Gemini / Groq reasoning, frame selection, token + cost tracking
│   ├── tools.py              # Forensic tools whose results are fed to the LLM
│   ├── reflection_agent.py   # Audits the LLM's reasoning, triggers a re-ask (max 2 passes)
│   ├── synthesis_agent.py    # Consensus weights, thresholds, final report
│   └── event_bus.py          # In-memory event store behind the SSE stream
├── preprocessing.py          # Validation, frame + burst extraction, metadata, C2PA, robustness score
├── main.py                   # Routes, auth, upload queue, background job
├── evaluate_pipeline.py      # Benchmark runner (accuracy, per-group rates, latency, cost)
├── download_benchmark.py     # Builds the 50-video public benchmark
├── export_generative_model.py# Optional: builds a local ONNX copy of the Swin detector
├── report.py, auth.py, models.py, schemas.py, database.py
├── config.py                 # Env settings, JWT secret handling, CPU thread cap read from the container's cgroup quota
├── model_onnx/               # INT8 face-swap ViT (83 MB)
frontend/src/components/
├── AuthLayout, Login, Register   # Landing page: pitch with an example report + the sign-in form
├── Upload                        # Intake page: list of checks + drop zone
├── StatusFeed                    # Live screen: progress bar per stage, agent graph, event stream, auto-opens the report
├── ResultsDashboard              # Verdict, score vs thresholds, time per stage
├── AgentBreakdown, ThresholdMeter, FrameGallery   # Five evidence cards in one row (Provenance, Visual, Temporal, Audio, LLM), score bars, frames with LLM notes
```

---

## Core Pipeline

```
 upload -> validate -> extract 6 keyframes (480p) + a short burst of consecutive frames after each
                                │
        ┌───────────┬───────────┼────────────┬──────────────────────┐
        ▼           ▼           ▼            ▼                      │ (background)
    [Visual]   [Temporal]   [Audio]     (provenance, read       [Generative]
   face-swap    motion +    spectral +    at validation)       AI-generation
      ViT       landmarks   lip-sync                              detector
        └───────────┴───────────┘                                   │
                    ▼                                               │
                 [Router]  picks 3-4 frames for the LLM             │
                    ▼                                               │
                  [LLM]  <──────┐  Gemini (Groq fallback)             │
                    ▼           │  with forensic tool results         │
              [Reflection] ─────┘  re-ask if reasoning contradicts    │
                    ▼                                               │
               [Synthesis] <────────────────────────────────────────┘
                    ▼
        verdict + confidence + per-frame explanations + report
```

The generative detector is optional and off unless `GEN_API_URL` (hosted) or a local model is configured; when on, it is the slowest CV step, so it runs in the background and is collected only at synthesis, hidden behind the LLM call (about 4-5 s).

### 1. Visual Forensics Agent (face-swap)
*   An INT8 Vision Transformer (`dima806/deepfake_vs_real_image_detection`) scores each frame. Input normalisation matches the model's own `preprocessor_config.json` (mean/std 0.5).
*   **Face gating:** MediaPipe checks each frame for a face. A face-swap classifier says nothing about a frame without a face, so frames with no face are excluded; if no frame has a face, the agent is `skipped` instead of voting "real".
*   Runs locally as ONNX (the deployed setup). If `GEN_API_URL` points at an external scoring service it runs there instead, and if that service is unreachable it degrades to noise/Laplacian heuristics and reports `fallback`.

### 2. Generative Video Agent (Sora / Veo / Kling / Wan), optional and off in the deployed configuration
*   A SwinV2 image classifier (`haywoodsloan/ai-image-detector-deploy`, 195M parameters) scores whole frames for generation artifacts. The video score is the mean of the more suspicious half of the frames.
*   Needs a model server (not included, not deployed). When used remotely, frames are sent as **lossless PNG** because JPEG recompression erases most of the signal (see Findings).

### 3. Temporal Consistency Agent
*   Each keyframe is followed by a burst of 8 consecutive frames, because flicker only shows between neighbouring frames, not one second apart.
*   **Motion spikes:** Farneback optical flow per consecutive pair; a spike is flagged relative to the burst's own typical motion. Scene cuts (histogram correlation) and duplicated frames are ignored.
*   **Landmark jitter:** MediaPipe FaceMesh rigid points, with head translation/rotation/zoom removed (similarity alignment), then the second difference measured as a fraction of eye distance (threshold 0.04; real faces measured at about 0.01, max 0.025).
*   Too few usable frames means `skipped`.

### 4. Audio Forensics Agent
*   Spectral cutoff (AI voice models roll off high frequencies), voice cadence (unnaturally flat energy) and lip-sync (mouth movement vs loudness). Silent videos are `skipped`.

### 5. Provenance Agent (C2PA Content Credentials)
*   Reads signed C2PA manifests with `c2pa-python`, including edit history. If the credentials declare the media generated by AI (`trainedAlgorithmicMedia`), confidence is forced to at least 0.95. Plain files without credentials are unaffected, so this catches only tools that embed credentials.
*   Filename heuristics (camera, social-media patterns) are shown as context but are not part of the verdict.

### 6. LLM Reasoning Agent (Gemini, Groq fallback)
*   Picks the 3-4 most suspicious frames (4 when the face-swap model couldn't run normally, otherwise 3; the minimum is 3 because the face-swap model's "clean" verdict says nothing about fully AI-generated video) (interleaving the face-swap model's flagged frames with temporal hits, then evenly spaced frames to fill), at 640 px.
*   Forensic tools (noise residual, landmark check, adjacent-frame comparison, metadata check) run on every selected frame and their findings are given to the model as observations. The model does not choose tools itself.
*   Returns a score plus a one or two sentence explanation per frame. Failed frames are excluded from the average rather than counted as "authentic". With no API key the agent is `skipped`.

### 7. Reflection Node
*   Audits the LLM's output for contradictions (high score with "authentic" language and the reverse, conflicting lighting or texture claims, missing explanations). If found, it loops back to the LLM with corrective feedback, at most 2 times.

### 8. Consensus Synthesis
*   **Base weights** (no audio / with audio): visual 45% / 35%, temporal 15%, LLM 40% / 35%, audio 15%.
*   **Adjustments:** a confident LLM (>0.60) takes 60-70%; visual `fallback` drops the visual weight to 10%; when the generative agent succeeds it takes 45% and the others are scaled down.
*   **Failed or skipped agents are dropped and the remaining weights renormalised.** No usable agent gives `UNCERTAIN`.
*   **Specialist boost:** if the visual, LLM or generative agent ran its model and scored at least 0.55, the final score is at least 0.85 times that score.
*   **Robustness damping:** low-resolution or low-bitrate videos (a way to hide artifacts) pull the score toward 0.5.
*   **Thresholds:** at least 0.48 `MANIPULATED`, below 0.30 `AUTHENTIC`, otherwise `UNCERTAIN`. `AUTHENTIC` additionally requires that the LLM ran, or that the generative detector ran and scored below 0.15, because the face-swap and motion checks cannot see fully AI-generated video.

---

## Evaluation

### Method
`python download_benchmark.py` builds a 50-video set from public Hugging Face datasets (seeded, so it is identical on every run, about 2 minutes, about 200 MB). `python evaluate_pipeline.py --dir benchmark --labels benchmark/labels.json` runs the same pipeline the web app uses on every video and writes `evaluation_scores.csv` with every agent's score.

| Group | n | Source |
|---|---|---|
| AI-generated: Sora 2 | 7 | `34data/gen-videos-sora2` |
| AI-generated: Veo 3 | 6 | `34data/gen-videos-veo3` |
| AI-generated: Kling | 6 | `34data/gen-videos-kling` |
| AI-generated: Wan 2 | 6 | `34data/gen-videos-wan2` |
| Real, HD (1080p) | 12 | OpenVid-1M (CC-BY-4.0) |
| Real, low-res (240p) | 13 | UCF101 subset |

### First live run (with the optional hosted detector, Gemini enabled)

| Metric | Result |
|---|---|
| Accuracy | **82.0%** (41 / 50) |
| Detection rate (recall) | **84.0%** (21 / 25 AI clips) |
| False-positive rate | **20.0%** (5 / 25 real clips) |
| Latency, median / 95th percentile | **5.5 s / 10.4 s** per video |
| Gemini calls per video | 1.08 |
| Tokens per video | about 5,100 input / 210 output |

| Group | Result |
|---|---|
| Sora 2 | 7 / 7 detected, **all via C2PA Content Credentials** (the pixel-level detectors caught 1 of 7) |
| Veo 3 | 6 / 6 detected (6 / 6 also without C2PA, so by pixel analysis) |
| Kling | 4 / 6 detected |
| Wan 2 | 4 / 6 detected |
| Real HD | 4 / 12 falsely flagged (33%) |
| Real low-res | 1 / 13 falsely flagged (8%) |

Cost per video is tokens multiplied by the provider's per-million-token prices. Set `GEMINI_INPUT_USD_PER_M` and `GEMINI_OUTPUT_USD_PER_M` and the script and the results page print it.

### Deployed configuration (live run, final code)
The deployed setup (no hosted AI-generation detector, face-swap model local, Gemini with a working Groq fallback, at least 3 frames to the LLM) was run live on all 50 videos:

| Metric | Result |
|---|---|
| Accuracy | **88%** (44 / 50) |
| Precision | **95%** (20 of the 21 videos flagged were AI) |
| Detection rate (recall) | **80%** (20 / 25 AI clips) |
| False-positive rate | **4%** (1 / 25 real clips) |
| Latency, median / 95th percentile | **3.4 s / 8.9 s** per video |
| LLM calls / tokens per video | 1.14 / about 5,400 input and 220 output |

By group: Sora 2 7/7, Veo 3 6/6, Kling 4/6, Wan 2 3/6; real HD 0 of 12 falsely flagged, real low-res 1 of 13 flagged and 4 of 13 answered UNCERTAIN.

How these were scored, and what they leave out:
*   **UNCERTAIN counts as "not flagged".** On real videos that makes it correct (4 clips here); on AI videos it would count as a miss (none occurred). If UNCERTAIN on a real video is counted as wrong, accuracy is 40 / 50 = **80%**, and the false-positive rate is unchanged.
*   **Confidence intervals (95%, Wilson):** accuracy 76-94%, precision 77-99%, recall 61-91%, false-positive rate 1-20%.
*   **C2PA did most of the work.** All 13 Sora 2 and Veo 3 clips carried a signed AI label, which is 13 of the 20 detections. The LLM's own score alone flags 13 of the 25 AI clips (Sora 2 1/7, Veo 3 5/6, Kling 4/6, Wan 2 3/6) and 0 of the 25 real ones. That is an estimate from the saved scores, not a re-run of the full consensus.
*   **The face-swap model ran on only 14 of the 50 videos** (the ones with a detected face). On the 3 AI clips it ran on it scored 0.18, 0.04 and 0.0, and it produced the one false positive.

Run-to-run variation: an earlier live run of the same configuration, before the Groq fallback was fixed (2 videos got no LLM verdict after Gemini rate limits), scored 86% accuracy, 91% precision and 8% false positives. Gemini's answers vary slightly between runs, so expect a point or two of movement.

### Ablation: which signal actually does the work?
From the first live run (with the optional detector and 2-4 frames to the LLM), re-running the consensus on its saved per-agent scores with parts removed:

| Configuration | Accuracy | Detection | False positives | Sora 2 | Veo 3 | Kling | Wan 2 | Real HD |
|---|---|---|---|---|---|---|---|---|
| Everything (hosted AI-generation detector + C2PA) | 82% | 21 / 25 | 5 / 25 | 7/7 | 6/6 | 4/6 | 4/6 | 4/12 flagged |
| Without the hosted AI-generation detector | **86%** | 20 / 25 | **2 / 25** | 7/7 | 6/6 | 4/6 | 3/6 | 1/12 flagged |
| Without C2PA | 70% | 15 / 25 | 5 / 25 | 1/7 | 6/6 | 4/6 | 4/6 | 4/12 flagged |
| Neither (face-swap ViT + temporal + audio + Gemini only) | 70% | 12 / 25 | 2 / 25 | 0/7 | 5/6 | 4/6 | 3/6 | 1/12 flagged |

Single signals on the 25 AI clips (score at least 0.5): face-swap ViT 0, Gemini 12, AI-generation detector 14, C2PA 13.

What this means:
*   **The face-swap ViT contributes nothing on fully AI-generated video** (0 of 25). It is built for swapped faces.
*   **All 7 Sora 2 detections came from C2PA credentials**, not from analysing pixels. Credentials survive when a file is downloaded straight from the generator, but social-media uploads and re-encoding strip them; the "Without C2PA" row is the realistic worst case.
*   **The hosted AI-generation detector did not help overall on this benchmark.** It caught one extra clip (Wan 2) but added three false alarms on real HD footage, so accuracy dropped from 86% to 82%. Its weight and threshold need tuning on a held-out split before it is worth the extra service.
*   Gemini on its own flags about half of the AI clips.

### Error analysis
First live run (the one with the optional detector):
*   **Missed AI clips (4):** two Kling and two Wan 2 clips where both the generative detector and Gemini scored low.
*   **False alarms on real HD (4):** three came from the generative detector, one from Gemini.
*   **False alarm on low-res (1):** the face-swap model on a makeup video.
*   **UNCERTAIN verdicts (3)** count as not manipulated.

Final run (deployed configuration), 6 errors in 50:
*   **Missed AI clips (5):** two Kling and three Wan 2 clips. Every agent called them real and the LLM scored each 0.05.
*   **False alarm (1):** a real low-resolution makeup clip from UCF101. The face-swap model scored 0.88 while the LLM scored 0.12; the consensus ended at 0.59.
*   **UNCERTAIN verdicts (4):** all on real low-resolution clips, counted as correct (see the scoring notes above).

### Limitations
*   **Small sample.** Fifty videos means roughly plus or minus 10 percentage points of uncertainty (see the confidence intervals above). Treat the numbers as an internal benchmark, not general accuracy.
*   **No held-out test set.** Prompts, frame selection and the decision to leave out the AI-generation detector were all made while looking at these same 50 videos, so the numbers are likely a little optimistic.
*   **Real and AI clips differ in more than realness.** The real low-res clips are 320x240 sports footage and the AI clips are HD cinematic scenes, so resolution and content type are mixed in with the label.
*   **Labels come from the datasets.** OpenVid-1M is scraped web footage, so a few "real" HD clips could be rendered or heavily processed.
*   **No face-swap videos in this benchmark.** Face-swap detection (Celeb-DF, FaceForensics++ style data) was not evaluated.
*   **Upload cap.** The evaluation script does not apply the web app's upload limit (about 1080p, 2.1 million pixels). The 1080p real clips fit it, but the 1440x1440 Kling clips (2.07 million pixels) are right at the edge, and anything larger would be rejected by the app.
*   **Latency conditions.** Measured on a laptop (the first run also had the optional detector running locally on 2 CPU threads); deployed numbers on Render's free CPU will differ.
*   **Provider mix not recorded.** The CSV records LLM calls and tokens but not whether Gemini or Groq answered. Gemini is the primary provider, so most verdicts come from it.
*   **The C2PA label is trusted without checking the signer.** An AI label forces the score to at least 0.95 even when its signature cannot be verified (the Sora 2 and Veo 3 clips here showed "signature could not be verified" and still triggered it). A forged label could therefore push a real video to MANIPULATED. A production version should require a valid signature from a trusted signer.
*   **C2PA depends on intact files.** Only some generators embed credentials, and re-encoding or social-media upload removes them. The Sora 2 and Veo 3 clips here were direct downloads with credentials intact, which flatters the overall numbers.
*   The dataset licenses vary (OpenVid-1M is CC-BY-4.0; the `34data` datasets do not state one). Use them for evaluation only and do not redistribute.

---

## Development Findings (what was tested and why the design looks like this)

| Finding | How it was measured | Decision |
|---|---|---|
| The face-swap ViT scores AI-generated clips about 0.00 | Ran it on Kling clips, full frames | Added a dedicated generative-video detector |
| Tight face crops make real faces look fake | 400 LFW photos: 89% flagged with crops vs 0% full-frame | Kept full-frame scoring; MediaPipe only gates face presence |
| JPEG recompression destroys the generative detector's signal | Same frames at q92: Kling scores fell from about 1.0 to 0.1-0.7 | Frames are sent to the scoring service as lossless PNG |
| Small detectors were not good enough | Smogy flagged 31% of real clips; small SWIN 15% (vs about 2% for the chosen model on 150 clips) | Chose the 195M-parameter Swin model for the optional detector |
| It cannot run on a 512 MB host | 199 MB INT8 file plus the stack exceeds memory | Prototyped hosting the models on a separate free service, then dropped it (see ablation); the deployed setup keeps the 83 MB face-swap model on Render |
| LangSmith tracing tripled memory (about 1.9 GB per job) and uploaded video frames | Server memory timeline with tracing on vs off | Removed; tracing is forced off in code |
| ONNX weight pre-packing duplicated weights | 259 MB vs 108 MB for the same model, identical outputs | Disabled pre-packing |
| The hosted service serialised requests | Per-stage profile: face-swap call waited behind the slow call | Sync endpoints so requests run concurrently |
| The old normalisation used ImageNet stats | Compared with `preprocessor_config.json` | Now uses mean/std 0.5 like the model |
| Noise thresholds never fired | Variance computed on a 0-255 scale against thresholds written for 0-1 | Same scale everywhere |
| Slow detector blocked the LLM | CV stage 1.1 s vs 0.35 s | Run in background, join at synthesis |

### Bugs fixed along the way
*   **Live agent steps never appeared in the UI.** LangGraph drops any state key that is not declared in the state type, and `job_id` was not declared, so every agent's progress event was silently thrown away and only "Job completed" arrived. Declaring it fixed the live stream.
*   The LLM's frame notes did not line up with the frames shown: Gemini sometimes returned keys like `1.333s` instead of `1.333`, and the gallery showed frames the LLM never saw. Keys are now normalised, and the gallery uses the exact timestamps that were sent to the LLM.
*   On Render the analysis took 49.5 s because of CPU thread oversubscription (see Production Safeguards).
*   Rate-limit handler returned HTTP 500 instead of 429.
*   Live-progress stream looped forever for unknown jobs; finished jobs' events are now pruned.
*   A failed or missing LLM was scored as "authentic"; it is now excluded or marked skipped.
*   All agents failing produced AUTHENTIC; it now produces UNCERTAIN.
*   WebM files with no frame count produced one frame instead of six.
*   `ffmpeg` and `ffprobe` calls had no timeouts.
*   The model could load twice (startup preload racing the first request).
*   `evaluate_pipeline.py` tested a different pipeline from production (re-encoded video, no audio).

---

## Production Safeguards for a 512 MB Host

*   **One analysis at a time** (`MAX_CONCURRENT_ANALYSES`, default 1); extra uploads wait with status "processing".
*   **Resolution cap** about 1080p (`MAX_VIDEO_PIXELS`, default 2,100,000) and a 30-second duration limit. Frames are decoded one at a time and shrunk to 480p as they arrive, so the cost is the decoder itself: measured at 720p +25 MB, 1080p +48 MB and 4K +167 MB. 4K does not fit, so it is rejected with a clear message.
*   **Memory fixes found by profiling:** ONNX weight pre-packing disabled (259 MB down to 108 MB for the same model), tracing removed (it copied every frame), PDF library imported lazily.
*   **Out-of-memory crashes fixed (measured per pipeline step with a memory sampler):**
    *   *An in-container re-encode was removed.* A change that accepted uploads up to 4K and re-encoded them to 480p with ffmpeg ran ffmpeg inside the same 512 MB container, adding 164 MB for 1080p and 487 MB for 4K on top of the server. That change caused the crashes.
    *   *Decoder threads.* OpenCV's FFmpeg decoder started one thread per visible core, each with its own frame buffers (the same problem as the CPU cap below). Decoding on one thread cut decode memory by 55-60% (1080p: 118 MB down to 48 MB).
    *   *ONNX Runtime arena disabled.* The arena keeps every buffer it ever allocated; without it, 55 MB less after inference, with identical outputs.
    *   *Memory handed back after each step.* Freed memory used to stay with the process, so every analysis started from a higher baseline. The server now calls `gc.collect()` and glibc's `malloc_trim` after frame extraction and after each job.
    *   *Result on macOS:* peak for a 720p job went from 558 MB to 457 MB, and for a 1080p job from 707 MB to 490 MB (idle about 340 MB). These are Mac figures and Linux numbers will differ. On Render, a 12.9 s, 964x1090 screen recording completed on the free 512 MB instance without running out of memory. Render's Metrics tab shows the real memory number, and `MALLOC_ARENA_MAX=2` helps there.
*   **Interrupted jobs.** If the server restarts mid-analysis (for example after running out of memory), any job still marked "processing" is marked failed at startup, so the page shows an error instead of waiting forever.
*   **CPU thread cap.** Render's free tier gives a fraction of a CPU but shows the container all of the host's cores. ONNX Runtime and OpenCV each started one thread per visible core, so the threads fought over a tiny share and the CV stage took 38.7 s (about 100x slower than the 0.4 s on a laptop; the whole analysis took 49.5 s). `config.py` now reads the container's cgroup CPU quota and caps all thread pools to it (`ANALYSIS_THREADS` overrides). A later re-run showed the CV stage at about 5 s. Hosted runs of a longer clip (12.9 s, 964x1090) still took 47-50 s in total: frame extraction 15.9 s, CV agents 22.9 s, LLM 7.0 s. The free instance has only a fraction of a core, so decoding and the CV agents are slow however the threads are set. This is a hosting limit, so it was left as is for a demo (on a laptop the same pipeline takes about 3.4 s).
*   **Gemini SDK imported at startup** instead of during the first analysis, which removed a multi-second delay on the first request after a cold start.

### Abuse protection (kept deliberately small for a demo)
*   **Rate limits** (`slowapi`): login 10 per minute, register 10 per hour, upload 5 per minute. On Render the limiter uses the visitor address from `X-Forwarded-For`; without that, every visitor would share the proxy's address and one limit.
*   **Queue cap:** at most 6 analyses queued or running (`MAX_PENDING_JOBS`). Every queued analysis holds a server worker thread, so without a cap a burst of uploads could freeze every other request.
*   **Upload hardening:** 50 MB limit while the file is saved, and the stored file name is a generated ID, so a client filename such as `../../x.mp4` cannot choose where the file goes. Only a cleaned display name is kept.
*   **Credentials:** passwords must be 8 to 72 bytes (bcrypt ignores or rejects anything longer). If `JWT_SECRET` is missing or is the old default that was published in this repository, the server signs tokens with a random secret instead, so a forged token is not possible; the cost is that everyone is signed out whenever the server restarts. Set `JWT_SECRET` in production.
*   **Not included:** account lockout, daily quotas, security headers, and limits on the live stream and PDF export. These were built and tested, then removed to keep the demo simple; they are the next things to add before real traffic. This also does not defend against a volumetric (network-level) attack, which needs a CDN or the host's own protection.

---

## Deployment (Render and Vercel)

**Render (backend), under Environment:**

| Variable | Value |
|---|---|
| `GEMINI_API_KEY` | primary LLM key |
| `GROQ_API_KEY` | fallback LLM key (without it, a Gemini failure means the LLM step is skipped) |
| `JWT_SECRET` | a long random value (for example `openssl rand -hex 32`) |
| `DATABASE_URL` | the Supabase Postgres connection string |
| `CORS_ORIGINS` | the Vercel site address |
| `MALLOC_ARENA_MAX` | `2` (lower memory use on Linux) |
| `ANALYSIS_THREADS` | `1` (suits the shared CPU) |

If a 1080p video still runs the instance out of memory, set `MAX_VIDEO_PIXELS=1000000` to limit uploads to about 720p.

**Vercel (frontend):** set `VITE_API_URL` to the Render address. `frontend/vercel.json` rewrites every path to `index.html`; without it, refreshing on `/login` or a report page returns "not found", because the site uses client-side routing.

---

## Setup

### Prerequisites
*   Python 3.10+ (3.9 works but disables the C2PA check), Node.js, FFmpeg/FFprobe on PATH (optional; without it audio forensics is skipped).

### Backend
```bash
cd backend
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload
```

`backend/.env` (all optional except the JWT secret):
```env
JWT_SECRET=long_random_value  # set this in production (for example: openssl rand -hex 32)
GEMINI_API_KEY=...            # primary LLM
GROQ_API_KEY=...              # fallback LLM
GROQ_VISION_MODEL=qwen/qwen3.8-27b  # optional override of the Groq model
DATABASE_URL=postgresql://... # Supabase; omit for local SQLite
GEN_API_URL=...               # optional external detector service; omit (deployed setup) to run the face-swap model locally
GEN_API_TOKEN=...             # bearer token for that service, if it needs one
GEMINI_INPUT_USD_PER_M=...    # per-million-token prices, to compute cost per video
GEMINI_OUTPUT_USD_PER_M=...
MAX_VIDEO_PIXELS=2100000      # upload cap (about 1080p)
MAX_CONCURRENT_ANALYSES=1
MAX_PENDING_JOBS=6            # queued + running analyses before uploads are refused
ANALYSIS_THREADS=...          # optional: override the CPU thread cap detected from the container
```

### Frontend
```bash
cd frontend && npm install && npm run dev
```
Set `VITE_API_URL` to the API address in production (Vercel).

### Reproduce the benchmark
```bash
pip install huggingface_hub remotezip
python download_benchmark.py
python evaluate_pipeline.py --dir benchmark --labels benchmark/labels.json
```

---

## Project Summary (resume bullets)

> **VeriFrame** | FastAPI, LangGraph, OpenCV, MediaPipe, ONNX Runtime, Gemini, Groq, React, Supabase · Render, Vercel

### Full version
*   Built a multi-agent deepfake detection system in LangGraph orchestrating ViT, OpenCV/MediaPipe temporal, audio and C2PA provenance agents, plus a Gemini vision agent with Groq fallback and self-correcting reflection loop, producing explainable per-frame verdicts streamed over SSE.
*   Benchmarked on 50 public videos from Sora 2, Veo 3, Kling, Wan 2 and real footage, achieving 88% accuracy, 95% precision, 80% recall, 4% false-positive rate.
*   Reduced median latency to 3.4s by running agents in parallel, downscaling frames to 480p, and routing only the 3–4 highest-risk frames to Gemini/Groq, averaging 1.14 LLM calls and ~5.4K tokens per video with per-stage latency and cost tracking in reports.
*   Engineered fault-tolerant consensus that redistributes agent weights on failure, automatically fails over from Gemini to Groq on rate limits, and returns UNCERTAIN when evidence is missing; cut ONNX model memory 259→108 MB to fit Render's 512 MB free tier instance.

### Short version
*   Built a LangGraph multi-agent deepfake detector fusing ViT, temporal, audio, C2PA and Gemini/Groq vision agents with a self-correcting reflection loop.
*   Achieved 88% accuracy, 95% precision and 4% false-positive rate on a 50-video benchmark spanning Sora 2, Veo 3, Kling and Wan 2.
*   Cut median latency to 3.4s via parallel agents and risk-based frame routing, using ~5.4K LLM tokens per video with Gemini→Groq failover.
