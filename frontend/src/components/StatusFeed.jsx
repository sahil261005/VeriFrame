import React, { useEffect, useState, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import { Film, ScanFace, Activity, AudioWaveform, GitBranch, BrainCircuit, RefreshCw, Scale, Check, X, LoaderCircle, Minus, TriangleAlert, Terminal } from 'lucide-react';
import { analysisService } from '../api';

// pipeline stages in graph order. `match` is a substring of the agent name the backend streams,
// `rank` orders the graph (visual/temporal/audio run in parallel), `weight` is the share of the progress bar
// (roughly how long the stage takes), `tau` is how fast the bar creeps while the stage is running (seconds)
const STAGES = [
  { key: 'ingest', name: 'Frame extraction', desc: 'Keyframes, motion bursts and C2PA check', icon: Film, rank: 0, weight: 8, tau: 3 },
  { key: 'visual', match: 'Visual Forensics', name: 'Visual agent', desc: 'Face-swap ViT on every keyframe', icon: ScanFace, rank: 1, weight: 12, tau: 6 },
  { key: 'temporal', match: 'Temporal Consistency', name: 'Temporal agent', desc: 'Optical flow and landmark jitter', icon: Activity, rank: 1, weight: 8, tau: 4 },
  { key: 'audio', match: 'Audio Forensics', name: 'Audio agent', desc: 'Spectral cutoff, cadence, lip-sync', icon: AudioWaveform, rank: 1, weight: 7, tau: 4 },
  { key: 'router', match: 'Conditional Router', name: 'Router', desc: 'Picks the riskiest frames for the LLM', icon: GitBranch, rank: 2, weight: 3, tau: 1 },
  { key: 'llm', match: 'Cognitive Reasoning', name: 'LLM reasoning', desc: 'Gemini (Groq fallback) explains each frame', icon: BrainCircuit, rank: 3, weight: 40, tau: 6 },
  { key: 'reflection', match: 'Reflection', name: 'Reflection', desc: 'Audits the LLM for contradictions', icon: RefreshCw, rank: 4, weight: 7, tau: 2 },
  { key: 'consensus', match: 'Consensus Engine', name: 'Consensus', desc: 'Weighted multi-agent verdict', icon: Scale, rank: 5, weight: 13, tau: 1 },
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
  for (const stage of STAGES) {
    const state = states[stage.key];
    if (state === 'done' || state === 'skipped' || state === 'failed') target += stage.weight;
    else if (state === 'active') {
      const first = stage.match ? events.find((e) => e.agent.includes(stage.match)) : null;
      const elapsed = Math.max(0, now - (first ? first.t : startTime)) / 1000;
      target += stage.weight * 0.9 * (1 - Math.exp(-elapsed / stage.tau));
    }
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

  const size = 220;
  const stroke = 12;
  const radius = (size - stroke) / 2;
  const circumference = 2 * Math.PI * radius;

  if (error) {
    return (
      <div className="card fade-up" style={{ maxWidth: '560px', margin: '48px auto 0', textAlign: 'center', padding: '36px' }}>
        <div className="dropzone-icon" style={{ background: 'var(--danger-soft)', color: 'var(--danger)', borderColor: 'rgba(248,113,113,0.35)' }}>
          <TriangleAlert size={26} />
        </div>
        <h2 style={{ fontSize: '20px', marginBottom: '6px' }}>Analysis failed</h2>
        <p style={{ color: 'var(--text-secondary)', marginBottom: '20px' }}>{error}</p>
        <button className="btn" onClick={() => onAnalysisComplete?.(jobId)}>Scan another video</button>
      </div>
    );
  }

  return (
    <div className="fade-up" style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '12px' }}>
        <div>
          <span className="eyebrow"><span className="live-dot" /> Real-Time LangGraph SSE Stream</span>
          <h2 style={{ fontSize: '24px', fontWeight: 800, letterSpacing: '-0.02em', marginTop: '12px' }}>
            {completed ? 'Analysis complete' : 'Running the multi-agent graph'}
          </h2>
        </div>
        <span className="chip mono">elapsed <strong>{elapsed}s</strong></span>
      </div>

      <div className="live-grid">
        <div className="card ring-wrap">
          <div className="ring">
            <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`} className={completed ? '' : 'ring-glow'}>
              <defs>
                <linearGradient id="ringGradient" x1="0" y1="0" x2="1" y2="1">
                  <stop offset="0%" stopColor="#6366f1" />
                  <stop offset="100%" stopColor="#22d3ee" />
                </linearGradient>
              </defs>
              <circle cx={size / 2} cy={size / 2} r={radius} fill="none" stroke="rgba(148,163,184,0.12)" strokeWidth={stroke} />
              <circle
                cx={size / 2}
                cy={size / 2}
                r={radius}
                fill="none"
                stroke={completed ? 'var(--success)' : 'url(#ringGradient)'}
                strokeWidth={stroke}
                strokeLinecap="round"
                strokeDasharray={circumference}
                strokeDashoffset={circumference * (1 - pct / 100)}
              />
            </svg>
            <div className="ring-center">
              <div className="ring-value">{Math.floor(pct)}<small>%</small></div>
              <div className="ring-label">{pct >= 100 ? 'complete' : completed ? 'finishing' : 'analyzing'}</div>
            </div>
          </div>
          <div style={{ minHeight: '42px' }}>
            <div style={{ fontWeight: 600, fontSize: '14px' }}>
              {completed ? 'Opening the forensic report…' : current ? current.name : 'Connecting…'}
            </div>
            <div className="muted" style={{ fontSize: '12.5px' }}>
              {completed ? 'Verdict ready' : current ? current.desc : 'Waiting for the first agent event'}
            </div>
          </div>
        </div>

        <div className="card">
          <div className="section-title"><GitBranch size={14} /> Agent graph</div>
          <div className="stepper">
            {STAGES.map((stage) => {
              const state = states[stage.key];
              const Icon = stage.icon;
              const StateIcon = { done: Check, skipped: Minus, failed: X }[state];
              return (
                <div key={stage.key} className={`step ${state}`}>
                  <div className="step-icon">
                    {state === 'active' ? <LoaderCircle size={17} style={{ animation: 'spin 1s linear infinite' }} /> : StateIcon ? <StateIcon size={17} /> : <Icon size={17} />}
                  </div>
                  <div>
                    <div className="step-name">{stage.name}</div>
                    <div className="step-desc">{stage.desc}</div>
                  </div>
                  <div className="step-state">{STATE_LABEL[state]}</div>
                </div>
              );
            })}
          </div>
        </div>
      </div>

      <div className="card">
        <div className="section-title"><Terminal size={14} /> Live event stream</div>
        <div className="terminal" ref={logRef}>
          <div className="terminal-bar">
            <i style={{ background: '#f87171' }} /><i style={{ background: '#fbbf24' }} /><i style={{ background: '#34d399' }} />
            <span style={{ marginLeft: '8px' }}>GET /stream/{jobId.slice(0, 8)}… · text/event-stream</span>
          </div>
          {events.length === 0 ? (
            <div className="muted">Connecting to the agent stream…</div>
          ) : (
            events.map((evt, idx) => (
              <div key={idx} className="log-line">
                <span className="log-time">+{((evt.t - startTime) / 1000).toFixed(1)}s</span>
                <span className="log-agent">{evt.agent}</span>
                <span className="log-msg">{evt.message}</span>
              </div>
            ))
          )}
        </div>
      </div>
    </div>
  );
}

export default StatusFeed;
