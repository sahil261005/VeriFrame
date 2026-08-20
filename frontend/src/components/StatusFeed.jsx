import React, { useEffect, useState, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import { analysisService } from '../api';

function StatusFeed({ jobId, onAnalysisComplete }) {
  const [status, setStatus] = useState('processing');
  const [error, setError] = useState('');
  const [liveEvents, setLiveEvents] = useState([]);
  const navigate = useNavigate();
  const eventsEndRef = useRef(null);

  // Auto-scroll to latest SSE event
  useEffect(() => {
    eventsEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [liveEvents]);

  useEffect(() => {
    let eventSource;
    let intervalId;

    // 1. Setup real-time SSE event stream
    eventSource = analysisService.createEventStream(
      jobId,
      (data) => {
        if (data.agent && data.message) {
          setLiveEvents((prev) => [...prev, data]);
        }
        if (data.status === 'completed') {
          setStatus('completed');
          onAnalysisComplete(jobId);
          navigate(`/analysis/${jobId}`);
        } else if (data.status === 'failed') {
          setStatus('failed');
          setError('Video analysis pipeline encountered an error.');
        }
      },
      () => {
        // If SSE fails or drops connection, fall back to traditional polling
        console.warn('SSE stream closed, falling back to polling...');
      }
    );

    // 2. Backup status polling (every 1s) to guarantee instantaneous completion redirect
    const checkStatus = async () => {
      try {
        const data = await analysisService.getAnalysis(jobId);
        setStatus(data.status);

        if (data.status === 'completed') {
          onAnalysisComplete(jobId);
          navigate(`/analysis/${jobId}`);
        } else if (data.status === 'failed') {
          setError('Video analysis pipeline encountered an error.');
        }
      } catch (err) {
        console.error('Error polling status fallback:', err);
      }
    };

    intervalId = setInterval(checkStatus, 1000);

    return () => {
      if (eventSource) eventSource.close();
      if (intervalId) clearInterval(intervalId);
    };
  }, [jobId, navigate, onAnalysisComplete]);

  const defaultSteps = [
    { agent: 'Visual Forensics', label: 'Spatial artifact & noise inspection' },
    { agent: 'Temporal Agent', label: 'Optical flow & facial landmark geometry tracking' },
    { agent: 'Audio Forensics', label: 'Spectral cutoff, voice cadence & lip-sync' },
    { agent: 'Router Node', label: 'Dynamic confidence routing (skip/extend LLM)' },
    { agent: 'Cognitive Reasoning', label: 'ReAct tool execution & LLM vision analysis' },
    { agent: 'Reflection Node', label: 'Self-correction & reasoning alignment audit' },
    { agent: 'Consensus Engine', label: 'Synthesizing final multi-agent verdict' }
  ];

  return (
    <div style={{ maxWidth: '680px', width: '100%', margin: '50px auto 0 auto' }}>
      <div className="card" style={{ padding: '36px', overflow: 'hidden' }}>
        <div style={{ textAlign: 'center', marginBottom: '30px' }}>
          {error ? (
            <div style={{ color: 'var(--danger)', fontSize: '18px', fontWeight: '600' }}>
              Analysis Failed
            </div>
          ) : (
            <>
              <div style={{ display: 'inline-flex', alignItems: 'center', gap: '8px', padding: '6px 14px', borderRadius: '20px', backgroundColor: 'rgba(59, 130, 246, 0.1)', color: 'var(--primary)', fontSize: '13px', fontWeight: '600', marginBottom: '12px' }}>
                <span style={{ width: '8px', height: '8px', borderRadius: '50%', backgroundColor: 'var(--primary)', animation: 'pulse 1.5s infinite' }} />
                Real-Time LangGraph SSE Stream
              </div>
              <h3 style={{ fontSize: '22px', fontWeight: '700', marginBottom: '6px' }}>
                Executing Multi-Agent Graph...
              </h3>
              <p style={{ fontSize: '14px', color: 'var(--text-secondary)' }}>
                LangGraph agents are streaming execution state via Server-Sent Events.
              </p>
            </>
          )}
        </div>

        {error ? (
          <div style={{ padding: '16px', borderRadius: '8px', backgroundColor: 'rgba(239, 68, 68, 0.1)', border: '1px solid rgba(239, 68, 68, 0.2)', color: 'var(--danger)', fontSize: '14px', textAlign: 'center' }}>
            {error}
          </div>
        ) : (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
            <h4 style={{ fontSize: '14px', fontWeight: '700', color: 'var(--text-primary)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
              Live Execution Feed:
            </h4>

            {/* Real-time SSE Events Log */}
            <div style={{ maxHeight: '280px', overflowY: 'auto', padding: '16px', borderRadius: '8px', backgroundColor: 'rgba(15, 23, 42, 0.05)', border: '1px solid var(--border-color)', display: 'flex', flexDirection: 'column', gap: '10px' }}>
              {liveEvents.length === 0 ? (
                <div style={{ fontSize: '13px', color: 'var(--text-secondary)', fontStyle: 'italic', textAlign: 'center', padding: '20px 0' }}>
                  Connecting to agent SSE stream...
                </div>
              ) : (
                liveEvents.map((evt, idx) => (
                  <div key={idx} style={{ display: 'flex', gap: '10px', alignItems: 'flex-start', fontSize: '13px' }}>
                    <span style={{ padding: '2px 8px', borderRadius: '4px', backgroundColor: 'rgba(59, 130, 246, 0.15)', color: 'var(--primary)', fontWeight: '600', fontSize: '11px', whiteSpace: 'nowrap', marginTop: '1px' }}>
                      {evt.agent}
                    </span>
                    <span style={{ color: 'var(--text-primary)', lineHeight: '1.4' }}>
                      {evt.message}
                    </span>
                  </div>
                ))
              )}
              <div ref={eventsEndRef} />
            </div>

            {/* Static Pipeline Checklist */}
            <div style={{ marginTop: '10px', display: 'flex', flexDirection: 'column', gap: '10px' }}>
              <div style={{ fontSize: '12px', fontWeight: '600', color: 'var(--text-secondary)', textTransform: 'uppercase' }}>
                Pipeline Graph Topology:
              </div>
              {defaultSteps.map((step, idx) => {
                const isExecuted = liveEvents.some((e) => e.agent && e.agent.toLowerCase().includes(step.agent.toLowerCase()));
                return (
                  <div key={idx} style={{ display: 'flex', alignItems: 'center', gap: '10px', fontSize: '13px' }}>
                    <span style={{ width: '16px', height: '16px', borderRadius: '50%', display: 'flex', alignItems: 'center', justifyContent: 'center', backgroundColor: isExecuted ? '#22c55e' : 'var(--border-color)', color: '#fff', fontSize: '10px', fontWeight: 'bold' }}>
                      {isExecuted ? '✓' : idx + 1}
                    </span>
                    <span style={{ fontWeight: '600', color: isExecuted ? 'var(--text-primary)' : 'var(--text-secondary)' }}>
                      {step.agent}:
                    </span>
                    <span style={{ color: 'var(--text-secondary)', fontSize: '12px' }}>
                      {step.label}
                    </span>
                  </div>
                );
              })}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default StatusFeed;
