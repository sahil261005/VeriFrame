import React, { useEffect, useState, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import { Film, ScanFace, Activity, AudioWaveform, GitBranch, BrainCircuit, RefreshCw, Scale, Check, X, LoaderCircle, Minus, TriangleAlert, Terminal } from 'lucide-react';
import { analysisService } from '../api';

// pipeline stages in graph order. `match` is a substring of the agent name the backend streams,
// `rank` orders the graph (visual/temporal/audio run in parallel), `weight` is the share of the progress bar
// (roughly how long the stage takes), `tau` is how fast the bar creeps while the stage is running (seconds)
const STAGES = [
  { key: 'ingest', short: 'frames', name: 'Frame extraction', desc: 'Keyframes, motion bursts and C2PA check', icon: Film, rank: 0, weight: 8, tau: 3 },
  { key: 'visual', short: 'visual', match: 'Visual Forensics', name: 'Visual agent', desc: 'Face-swap ViT on every keyframe', icon: ScanFace, rank: 1, weight: 12, tau: 6 },
  { key: 'temporal', short: 'motion', match: 'Temporal Consistency', name: 'Temporal agent', desc: 'Optical flow and landmark jitter', icon: Activity, rank: 1, weight: 8, tau: 4 },
  { key: 'audio', short: 'audio', match: 'Audio Forensics', name: 'Audio agent', desc: 'Spectral cutoff, cadence, lip-sync', icon: AudioWaveform, rank: 1, weight: 7, tau: 4 },
  { key: 'router', short: 'route', match: 'Conditional Router', name: 'Router', desc: 'Picks the riskiest frames for the LLM', icon: GitBranch, rank: 2, weight: 3, tau: 1 },
  { key: 'llm', short: 'LLM reasoning', match: 'Cognitive Reasoning', name: 'LLM reasoning', desc: 'Gemini (Groq fallback) explains each frame', icon: BrainCircuit, rank: 3, weight: 40, tau: 6 },
  { key: 'reflection', short: 'reflect', match: 'Reflection', name: 'Reflection', desc: 'Audits the LLM for contradictions', icon: RefreshCw, rank: 4, weight: 7, tau: 2 },
  { key: 'consensus', short: 'verdict', match: 'Consensus Engine', name: 'Consensus', desc: 'Weighted multi-agent verdict', icon: Scale, rank: 5, weight: 13, tau: 1 },
];

const finalState = (message) => {
  if (/^(skipped|no audio)/i.test(message)) return 'skipped';
  if (/^failed|synthesis failed/i.test(message)) return 'failed';
  if (/^completed|passed|^verdict rendered/i.test(message)) return 'done';
  return null;
};

function stageStates(events, completed) {
  const states = {};
  for (const stage of STAGES) {
    if (!stage.match) continue;
    const own = events.filter((e) => e.agent.includes(stage.match));
    if (own.length === 0) states[stage.key] = 'pending';
    else if (stage.key === 'router') states[stage.key] = 'done';
    else states[stage.key] = finalState(own[own.length - 1].message) || 'active';
  }
  states.ingest = events.length > 0 ? 'done' : 'active';

  // a stage further down the graph has started, so everything before it has finished
  const maxRank = Math.max(0, ...STAGES.filter((s) => states[s.key] !== 'pending').map((s) => s.rank));
  for (const stage of STAGES) {
    if (stage.rank < maxRank && (states[stage.key] === 'pending' || states[stage.key] === 'active')) {
      states[stage.key] = 'done';
    }
    if (completed && (states[stage.key] === 'pending' || states[stage.key] === 'active')) {
      states[stage.key] = 'done';
    }
  }

  // the reflection loop can send the graph back to the LLM: whatever stage spoke last and has not
  // finished is the one running now, and anything after it in the graph has finished its pass
  const last = events[events.length - 1];
  const lastStage = last && STAGES.find((s) => s.match && last.agent.includes(s.match));
  if (!completed && lastStage && !finalState(last.message) && lastStage.key !== 'router') {
    states[lastStage.key] = 'active';
    for (const stage of STAGES) {
      if (stage.rank > lastStage.rank && states[stage.key] === 'active') states[stage.key] = 'done';
    }
  }
  return states;
}

const STATE_LABEL = { pending: 'queued', active: 'running', done: 'done', skipped: 'skipped', failed: 'failed' };

function StatusFeed({ jobId, onAnalysisComplete }) {
  const [error, setError] = useState('');
  const [events, setEvents] = useState([]);
  const [completed, setCompleted] = useState(false);
  const [display, setDisplay] = useState(0);
  const [startTime] = useState(() => Date.now());
  const [now, setNow] = useState(startTime);
  const navigate = useNavigate();

  const targetRef = useRef(0);
  const finishedRef = useRef(false);
  const completeCb = useRef(onAnalysisComplete);
  const logRef = useRef(null);

  useEffect(() => {
    completeCb.current = onAnalysisComplete;
  }, [onAnalysisComplete]);

  // SSE stream plus a 1s status poll as a safety net (SSE can drop on free hosting)
  useEffect(() => {
    const markDone = () => setCompleted(true);
    const markFailed = () => setError('The analysis pipeline hit an error. Please try again.');

    const eventSource = analysisService.createEventStream(
      jobId,
      (data) => {
        if (data.agent && data.message) {
          setEvents((prev) => [...prev, { agent: data.agent, message: data.message, t: Date.now() }]);
        }
        if (data.status === 'completed') markDone();
        else if (data.status === 'failed') markFailed();
      },
      () => console.warn('SSE stream closed, relying on status polling')
    );

    const poll = setInterval(async () => {
      try {
        const data = await analysisService.getAnalysis(jobId);
        if (data.status === 'completed') markDone();
        else if (data.status === 'failed') markFailed();
      } catch (err) {
        console.error('status poll failed:', err);
      }
    }, 1000);

    return () => {
      eventSource?.close();
      clearInterval(poll);
    };
  }, [jobId]);

  const states = stageStates(events, completed);

  // progress target: finished stages count fully, running stages creep towards 90% of their share
  // (a stage is running since its first streamed event; frame extraction since the page opened)
  let target = 0;
  const stageFill = {};
  for (const stage of STAGES) {
    const state = states[stage.key];
    let fill = 0;
    if (state === 'done' || state === 'skipped' || state === 'failed') fill = 1;
    else if (state === 'active') {
      const first = stage.match ? events.find((e) => e.agent.includes(stage.match)) : null;
      const elapsed = Math.max(0, now - (first ? first.t : startTime)) / 1000;
      fill = 0.9 * (1 - Math.exp(-elapsed / stage.tau));
    }
    stageFill[stage.key] = fill;
    target += stage.weight * fill;
  }
  target = completed ? 100 : Math.min(target, 98);

  useEffect(() => {
    targetRef.current = target;
  }, [target]);

  // animation tick: ease the shown number toward the target, never backwards
  useEffect(() => {
    const id = setInterval(() => {
      setNow(Date.now());
      setDisplay((d) => {
        const t = targetRef.current;
        const next = d + Math.max(0, t - d) * (t === 100 ? 0.3 : 0.12);
        return t === 100 && 100 - next < 0.4 ? 100 : next;
      });
    }, 80);
    return () => clearInterval(id);
  }, []);

  // as soon as the bar reaches 100, open the report
  useEffect(() => {
    if (display >= 100 && completed && !finishedRef.current) {
      finishedRef.current = true;
      const id = setTimeout(() => {
        completeCb.current?.(jobId);
        navigate(`/analysis/${jobId}`);
      }, 450);
      return () => clearTimeout(id);
    }
  }, [display, completed, jobId, navigate]);

  // keep the newest event in view inside the log box (scrolling the box, not the page)
  useEffect(() => {
    const el = logRef.current;
    if (el) el.scrollTo({ top: el.scrollHeight, behavior: 'smooth' });
  }, [events]);

  const pct = Math.min(100, display);
  const current = [...STAGES].reverse().find((s) => states[s.key] === 'active');
  const elapsed = ((now - startTime) / 1000).toFixed(1);

  if (error) {
    return (
      <div className="card" style={{ maxWidth: '560px', margin: '48px auto 0', padding: '32px' }}>
        <span className="badge badge-manipulated"><TriangleAlert size={13} /> Failed</span>
        <h1 style={{ fontSize: '26px', margin: '14px 0 6px' }}>The analysis did not finish</h1>
        <p style={{ color: 'var(--ink-2)', marginBottom: '20px' }}>{error}</p>
        <button className="btn" onClick={() => onAnalysisComplete?.(jobId)}>Upload another video</button>
      </div>
    );
  }

  return (
    <div>
      <div className="run-head">
        <div>
          <span className="label live-pill">{!completed && <span className="live-dot" />}Live LangGraph stream · case {jobId.slice(0, 8)}</span>
          <h1>{completed ? 'Analysis complete' : 'Analyzing your video'}</h1>
        </div>
        <span className="mono muted" style={{ fontSize: '13px' }}>elapsed {elapsed}s</span>
      </div>

      <div className="card progress-sheet">
        <div className="progress-top">
          <div className="progress-pct">{Math.floor(pct)}<small>%</small></div>
          <div className="progress-now">
            <span className="label">{completed ? 'done' : 'now running'}</span>
            <strong>{completed ? 'Opening the report…' : current ? current.name : 'Connecting…'}</strong>
            <span className="muted" style={{ fontSize: '13px' }}>{completed ? 'Verdict ready' : current ? current.desc : 'Waiting for the first agent event'}</span>
          </div>
        </div>
        <div className="segbar" role="progressbar" aria-valuenow={Math.floor(pct)} aria-valuemin={0} aria-valuemax={100}>
          {STAGES.map((stage) => (
            <div key={stage.key} className={`seg ${states[stage.key]}`} style={{ flex: stage.weight }}>
              <div className="seg-fill" style={{ width: `${(completed ? 1 : stageFill[stage.key]) * 100}%` }} />
            </div>
          ))}
        </div>
        <div className="seg-labels">
          {STAGES.map((stage) => <span key={stage.key} style={{ flex: stage.weight }}>{stage.short}</span>)}
        </div>
      </div>

      <div className="run-grid">
        <div className="card">
          <div className="section-title"><h2>Agent graph</h2><span className="label">{Object.values(states).filter((v) => v !== 'pending' && v !== 'active').length} / {STAGES.length} finished</span></div>
          <div className="stepper">
            {STAGES.map((stage) => {
              const state = states[stage.key];
              const Icon = stage.icon;
              const StateIcon = { done: Check, skipped: Minus, failed: X }[state];
              return (
                <div key={stage.key} className={`step ${state}`}>
                  <div className="step-icon">
                    {state === 'active' ? <LoaderCircle size={15} style={{ animation: 'spin 1s linear infinite' }} /> : StateIcon ? <StateIcon size={15} /> : <Icon size={15} />}
                  </div>
                  <div style={{ minWidth: 0 }}>
                    <div className="step-name">{stage.name}</div>
                    <div className="step-desc">{stage.desc}</div>
                  </div>
                  <div className="step-state">{STATE_LABEL[state]}</div>
                </div>
              );
            })}
          </div>
        </div>

        <div className="card">
          <div className="section-title"><h2>Event stream</h2><span className="label"><Terminal size={12} style={{ verticalAlign: '-2px' }} /> text/event-stream</span></div>
          <div className="terminal" ref={logRef}>
            {events.length === 0 ? (
              <div className="muted">Connecting to the agent stream…</div>
            ) : (
              events.map((evt, idx) => (
                <div key={idx} className="log-line">
                  <span className="log-time">+{((evt.t - startTime) / 1000).toFixed(1)}s</span>
                  <span><span className="log-agent">{evt.agent}</span><span className="log-msg">{evt.message}</span></span>
                </div>
              ))
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

export default StatusFeed;
