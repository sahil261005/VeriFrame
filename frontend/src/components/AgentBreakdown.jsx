import React from 'react';
import { Camera, Cpu, Brain, AlertTriangle } from 'lucide-react';

function AgentBreakdown({ breakdown, isPartial }) {
  const visual = breakdown?.visual_agent || {};
  const temporal = breakdown?.temporal_agent || {};
  const llm = breakdown?.llm_agent || {};

  return (
    <div style={{ width: '100%', display: 'flex', flexDirection: 'column', gap: '20px' }}>
      {isPartial && (
        <div style={{
          padding: '16px',
          borderRadius: '8px',
          backgroundColor: 'var(--warning-glow)',
          border: '1px solid rgba(245, 158, 11, 0.2)',
          color: 'var(--warning)',
          fontSize: '14px',
          display: 'flex',
          gap: '10px',
          alignItems: 'center'
        }}>
          <AlertTriangle size={20} style={{ flexShrink: 0 }} />
          <div>
            <strong>Partial Analysis Warning:</strong> Some analysis agents failed during execution. Consensus weights were dynamically redistributed among available agents.
          </div>
        </div>
      )}

      <div style={{
        display: 'grid',
        gridTemplateColumns: 'repeat(auto-fit, minmax(250px, 1fr))',
        gap: '20px'
      }}>
        {/* Visual Card */}
        <div className="card" style={{ opacity: visual.status === 'failed' ? 0.5 : 1 }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '16px' }}>
            <div style={{
              width: '40px',
              height: '40px',
              borderRadius: '8px',
              backgroundColor: visual.status === 'failed' ? 'rgba(239, 68, 68, 0.1)' : 'rgba(59, 130, 246, 0.1)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              color: visual.status === 'failed' ? 'var(--danger)' : 'var(--primary)'
            }}>
              <Camera size={22} />
            </div>
            <span className={`badge ${visual.status === 'success' ? 'badge-authentic' : 'badge-manipulated'}`} style={{ border: 'none' }}>
              {visual.status}
            </span>
          </div>
          <h4 style={{ fontSize: '16px', fontWeight: '700', marginBottom: '6px' }}>Visual Forensics</h4>
          <p style={{ fontSize: '13px', color: 'var(--text-secondary)', marginBottom: '16px' }}>
            Inspects spatial face classification and frame noise consistency.
          </p>
          <div style={{ borderTop: '1px solid var(--border-color)', paddingTop: '12px', display: 'flex', justifyContent: 'space-between', fontSize: '14px' }}>
            <span style={{ color: 'var(--text-secondary)' }}>Fake Rating</span>
            <strong style={{ color: 'var(--text-primary)' }}>{visual.status === 'failed' ? '0.00' : (visual.score || 0.0).toFixed(2)}</strong>
          </div>
        </div>

        {/* Temporal Card */}
        <div className="card" style={{ opacity: temporal.status === 'failed' ? 0.5 : 1 }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '16px' }}>
            <div style={{
              width: '40px',
              height: '40px',
              borderRadius: '8px',
              backgroundColor: temporal.status === 'failed' ? 'rgba(239, 68, 68, 0.1)' : 'rgba(245, 158, 11, 0.1)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              color: temporal.status === 'failed' ? 'var(--danger)' : 'var(--warning)'
            }}>
              <Cpu size={22} />
            </div>
            <span className={`badge ${temporal.status === 'success' ? 'badge-authentic' : 'badge-manipulated'}`} style={{ border: 'none' }}>
              {temporal.status}
            </span>
          </div>
          <h4 style={{ fontSize: '16px', fontWeight: '700', marginBottom: '6px' }}>Temporal Consistency</h4>
          <p style={{ fontSize: '13px', color: 'var(--text-secondary)', marginBottom: '16px' }}>
            Analyzes dense optical flow motion and landmark geometry shifts.
          </p>
          <div style={{ borderTop: '1px solid var(--border-color)', paddingTop: '12px', display: 'flex', justifyContent: 'space-between', fontSize: '14px' }}>
            <span style={{ color: 'var(--text-secondary)' }}>Anomaly Score</span>
            <strong style={{ color: 'var(--text-primary)' }}>{temporal.status === 'failed' ? '0.00' : (temporal.score || 0.0).toFixed(2)}</strong>
          </div>
        </div>

        {/* LLM Card */}
        <div className="card" style={{ opacity: llm.status === 'failed' ? 0.5 : 1 }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '16px' }}>
            <div style={{
              width: '40px',
              height: '40px',
              borderRadius: '8px',
              backgroundColor: llm.status === 'failed' ? 'rgba(239, 68, 68, 0.1)' : 'rgba(16, 185, 129, 0.1)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              color: llm.status === 'failed' ? 'var(--danger)' : 'var(--success)'
            }}>
              <Brain size={22} />
            </div>
            <span className={`badge ${llm.status === 'success' ? 'badge-authentic' : 'badge-manipulated'}`} style={{ border: 'none' }}>
              {llm.status}
            </span>
          </div>
          <h4 style={{ fontSize: '16px', fontWeight: '700', marginBottom: '6px' }}>AI Reasoning Swarm</h4>
          <p style={{ fontSize: '13px', color: 'var(--text-secondary)', marginBottom: '16px' }}>
            Calls Gemini 2.5 Flash to inspect anomalies on target frames.
          </p>
          <div style={{ borderTop: '1px solid var(--border-color)', paddingTop: '12px', display: 'flex', justifyContent: 'space-between', fontSize: '14px' }}>
            <span style={{ color: 'var(--text-secondary)' }}>Average Score</span>
            <strong style={{ color: 'var(--text-primary)' }}>{llm.status === 'failed' ? '0.00' : (llm.score || 0.0).toFixed(2)}</strong>
          </div>
        </div>
      </div>
    </div>
  );
}

export default AgentBreakdown;
