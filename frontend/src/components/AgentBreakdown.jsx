import React from 'react';

function AgentBreakdown({ breakdown, isPartial }) {
  const visual = breakdown?.visual_agent || {};
  const temporal = breakdown?.temporal_agent || {};
  const audio = breakdown?.audio_agent || {};
  const generative = breakdown?.generative_agent || {};
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

      {provenance.c2pa_present && (
        <div style={{
          padding: '16px',
          borderRadius: '8px',
          border: `1px solid ${provenance.c2pa_ai_generated ? 'rgba(239, 68, 68, 0.3)' : 'rgba(16, 185, 129, 0.3)'}`,
          color: provenance.c2pa_ai_generated ? 'var(--danger, #ef4444)' : 'var(--success)',
          fontSize: '14px'
        }}>
          <strong>Content Credentials (C2PA) found:</strong>{' '}
          {provenance.c2pa_ai_generated
            ? `the file's signed metadata declares it AI-generated${provenance.c2pa_generator ? ` by ${provenance.c2pa_generator}` : ''}.`
            : `signed by ${provenance.c2pa_generator || 'an unknown tool'}, with no AI-generation declared.`}
          {!provenance.c2pa_signature_valid && ' (signature could not be verified)'}
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
            <span className={`badge ${visual.status === 'success' ? 'badge-authentic' : visual.status === 'fallback' || visual.status === 'skipped' ? 'badge-uncertain' : 'badge-manipulated'}`} style={{ border: 'none' }}>
              {visual.status === 'fallback' ? 'heuristics' : visual.status === 'skipped' ? 'No Face' : visual.status}
            </span>
          </div>
          <h4 style={{ fontSize: '16px', fontWeight: '700', marginBottom: '6px' }}>Visual Forensics</h4>
          <p style={{ fontSize: '13px', color: 'var(--text-secondary)', marginBottom: '16px' }}>
            Inspects spatial face classification and frame noise consistency.
          </p>
          <div style={{ borderTop: '1px solid var(--border-color)', paddingTop: '12px', display: 'flex', justifyContent: 'space-between', fontSize: '14px' }}>
            <span style={{ color: 'var(--text-secondary)' }}>Fake Rating</span>
            <strong style={{ color: 'var(--text-primary)' }}>{visual.status === 'failed' ? '0.00' : visual.status === 'skipped' ? 'N/A' : (visual.score || 0.0).toFixed(2)}</strong>
          </div>
        </div>

        {/* Temporal Card */}
        <div className="card" style={{ opacity: temporal.status === 'failed' ? 0.5 : 1 }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}>
            <span style={{ fontSize: '12px', fontWeight: 'bold', textTransform: 'uppercase', color: 'var(--warning)' }}>[ Temporal ]</span>
            <span className={`badge ${temporal.status === 'success' ? 'badge-authentic' : temporal.status === 'skipped' ? 'badge-uncertain' : 'badge-manipulated'}`} style={{ border: 'none' }}>
              {temporal.status}
            </span>
          </div>
          <h4 style={{ fontSize: '16px', fontWeight: '700', marginBottom: '6px' }}>Temporal Consistency</h4>
          <p style={{ fontSize: '13px', color: 'var(--text-secondary)', marginBottom: '16px' }}>
            Checks consecutive frames for motion spikes and facial landmark jitter.
          </p>
          <div style={{ borderTop: '1px solid var(--border-color)', paddingTop: '12px', display: 'flex', justifyContent: 'space-between', fontSize: '14px' }}>
            <span style={{ color: 'var(--text-secondary)' }}>Anomaly Score</span>
            <strong style={{ color: 'var(--text-primary)' }}>{temporal.status === 'failed' ? '0.00' : temporal.status === 'skipped' ? 'N/A' : (temporal.score || 0.0).toFixed(2)}</strong>
          </div>
        </div>

        {/* Audio Card (Replaces Provenance) */}
        <div className="card" style={{ opacity: audio.status === 'failed' ? 0.5 : 1 }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}>
            <span style={{ fontSize: '12px', fontWeight: 'bold', textTransform: 'uppercase', color: '#ec4899' }}>[ Acoustic ]</span>
            <span className={`badge ${audio.status === 'success' ? 'badge-authentic' : audio.status === 'skipped' ? 'badge-uncertain' : 'badge-manipulated'}`} style={{ border: 'none' }}>
              {audio.status === 'skipped' ? 'Silent Video' : audio.status || 'Active'}
            </span>
          </div>
          <h4 style={{ fontSize: '16px', fontWeight: '700', marginBottom: '6px' }}>Audio & Lip-Sync</h4>
          <p style={{ fontSize: '13px', color: 'var(--text-secondary)', marginBottom: '16px' }}>
            {audio.has_audio ? (audio.details?.summary || 'Analyzes spectral cutoffs & mouth-voice sync.') : 'No audio track detected in media.'}
          </p>
          <div style={{ borderTop: '1px solid var(--border-color)', paddingTop: '12px', display: 'flex', justifyContent: 'space-between', fontSize: '14px' }}>
            <span style={{ color: 'var(--text-secondary)' }}>Audio Fake Score</span>
            <strong style={{ color: 'var(--text-primary)' }}>{audio.has_audio ? (audio.score || 0.0).toFixed(2) : 'N/A'}</strong>
          </div>
        </div>

        {/* Generative Video Card */}
        <div className="card" style={{ opacity: generative.status === 'failed' ? 0.5 : 1 }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}>
            <span style={{ fontSize: '12px', fontWeight: 'bold', textTransform: 'uppercase', color: '#8b5cf6' }}>[ Generative ]</span>
            <span className={`badge ${generative.status === 'success' ? 'badge-authentic' : generative.status === 'skipped' ? 'badge-uncertain' : 'badge-manipulated'}`} style={{ border: 'none' }}>
              {generative.status === 'skipped' ? 'Not Installed' : generative.status}
            </span>
          </div>
          <h4 style={{ fontSize: '16px', fontWeight: '700', marginBottom: '6px' }}>AI-Generated Video</h4>
          <p style={{ fontSize: '13px', color: 'var(--text-secondary)', marginBottom: '16px' }}>
            Scans whole frames for artifacts of AI video models such as Sora, Kling, Runway and Veo.
          </p>
          <div style={{ borderTop: '1px solid var(--border-color)', paddingTop: '12px', display: 'flex', justifyContent: 'space-between', fontSize: '14px' }}>
            <span style={{ color: 'var(--text-secondary)' }}>AI-Generated Score</span>
            <strong style={{ color: 'var(--text-primary)' }}>{generative.status === 'success' ? (generative.score || 0.0).toFixed(2) : 'N/A'}</strong>
          </div>
        </div>

        {/* LLM Card */}
        <div className="card" style={{ opacity: llm.status === 'failed' ? 0.5 : 1 }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}>
            <span style={{ fontSize: '12px', fontWeight: 'bold', textTransform: 'uppercase', color: 'var(--success)' }}>[ Semantic ]</span>
            <span className={`badge ${llm.status === 'success' ? 'badge-authentic' : llm.status === 'skipped' ? 'badge-uncertain' : 'badge-manipulated'}`} style={{ border: 'none' }}>
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
      </div>
    </div>
  );
}

export default AgentBreakdown;


