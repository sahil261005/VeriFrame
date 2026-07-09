import React from 'react';

function AgentBreakdown({ breakdown, isPartial }) {
  const visual = breakdown?.visual_agent || {};
  const temporal = breakdown?.temporal_agent || {};
  const llm = breakdown?.llm_agent || {};
  const provenance = breakdown?.provenance_agent || {};

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
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}>
            <span style={{ fontSize: '12px', fontWeight: 'bold', textTransform: 'uppercase', color: 'var(--primary)' }}>[ Visual ]</span>
            <span className={`badge ${visual.status === 'success' ? 'badge-authentic' : visual.status === 'fallback' ? 'badge-uncertain' : 'badge-manipulated'}`} style={{ border: 'none' }}>
              {visual.status === 'fallback' ? 'heuristics' : visual.status}
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
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}>
            <span style={{ fontSize: '12px', fontWeight: 'bold', textTransform: 'uppercase', color: 'var(--warning)' }}>[ Temporal ]</span>
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
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}>
            <span style={{ fontSize: '12px', fontWeight: 'bold', textTransform: 'uppercase', color: 'var(--success)' }}>[ Semantic ]</span>
            <span className={`badge ${llm.status === 'success' ? 'badge-authentic' : 'badge-manipulated'}`} style={{ border: 'none' }}>
              {llm.status}
            </span>
          </div>
          <h4 style={{ fontSize: '16px', fontWeight: '700', marginBottom: '6px' }}>Semantic Coherence</h4>
          <p style={{ fontSize: '13px', color: 'var(--text-secondary)', marginBottom: '16px' }}>
            Evaluates contextual sync and high-level scene consistency.
          </p>
          <div style={{ borderTop: '1px solid var(--border-color)', paddingTop: '12px', display: 'flex', justifyContent: 'space-between', fontSize: '14px' }}>
            <span style={{ color: 'var(--text-secondary)' }}>Average Score</span>
            <strong style={{ color: 'var(--text-primary)' }}>{llm.status === 'failed' ? '0.00' : (llm.score || 0.0).toFixed(2)}</strong>
          </div>
        </div>

        {/* Provenance Card */}
        <div className="card" style={{ opacity: provenance.status === 'failed' ? 0.5 : 1 }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}>
            <span style={{ fontSize: '12px', fontWeight: 'bold', textTransform: 'uppercase', color: 'var(--primary)' }}>[ Provenance ]</span>
            <span className={`badge ${provenance.c2pa_compliant ? 'badge-authentic' : 'badge-uncertain'}`} style={{ border: 'none' }}>
              {provenance.c2pa_compliant ? 'C2PA Compliant' : 'No C2PA'}
            </span>
          </div>
          <h4 style={{ fontSize: '16px', fontWeight: '700', marginBottom: '6px' }}>Provenance & C2PA Lineage</h4>
          <p style={{ fontSize: '13px', color: 'var(--text-secondary)', marginBottom: '12px' }}>
            Scans file structure for cryptographic origin stamps and metadata integrity.
          </p>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '6px', fontSize: '13px', marginBottom: '12px', borderTop: '1px solid var(--border-color)', paddingTop: '12px' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between' }}>
              <span style={{ color: 'var(--text-secondary)' }}>Metadata Stripped</span>
              <span style={{ color: provenance.metadata_stripped ? 'var(--danger)' : 'var(--success)', fontWeight: '600' }}>
                {provenance.metadata_stripped ? 'Yes' : 'No'}
              </span>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between' }}>
              <span style={{ color: 'var(--text-secondary)' }}>Encoder</span>
              <span style={{ color: 'var(--text-primary)', fontWeight: '600', textTransform: 'capitalize' }}>
                {provenance.encoder || 'unknown'}
              </span>
            </div>
          </div>
          <div style={{ borderTop: '1px solid var(--border-color)', paddingTop: '12px', display: 'flex', justifyContent: 'space-between', fontSize: '14px' }}>
            <span style={{ color: 'var(--text-secondary)' }}>Provenance Score</span>
            <strong style={{ color: 'var(--text-primary)' }}>{(provenance.score || 0.5).toFixed(2)}</strong>
          </div>
        </div>
      </div>
    </div>
  );
}

export default AgentBreakdown;


