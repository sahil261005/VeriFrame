import React from 'react';
import { ScanFace, Activity, AudioWaveform, BrainCircuit, BadgeCheck, TriangleAlert, ShieldAlert, Layers } from 'lucide-react';

const STATUS_BADGE = {
  success: ['badge-authentic', 'Ran'],
  fallback: ['badge-uncertain', 'Heuristics'],
  skipped: ['badge-neutral', 'Skipped'],
  failed: ['badge-manipulated', 'Failed'],
};

// 0 = looks real (green) ... 1 = looks fake (red)
const scoreColor = (s) => (s >= 0.48 ? 'var(--danger)' : s >= 0.3 ? 'var(--warning)' : 'var(--success)');

function AgentCard({ icon: Icon, accent, title, status, badgeText, description, scoreLabel, score, footnote }) {
  const [badgeClass, defaultText] = STATUS_BADGE[status] || ['badge-neutral', status || 'n/a'];
  const hasScore = typeof score === 'number';
  return (
    <div className="card agent-card" style={{ opacity: status === 'failed' ? 0.6 : 1 }}>
      <div className="agent-head">
        <div className="agent-icon" style={{ color: accent, background: `color-mix(in srgb, ${accent} 12%, transparent)` }}>
          <Icon size={19} />
        </div>
        <span className={`badge ${badgeClass}`}>{badgeText || defaultText}</span>
      </div>
      <div className="agent-name">{title}</div>
      <div className="agent-desc" title={typeof description === 'string' ? description : undefined}>
        <span style={{ display: '-webkit-box', WebkitLineClamp: 5, WebkitBoxOrient: 'vertical', overflow: 'hidden' }}>{description}</span>
      </div>
      <div className="score-row">
        <span>{scoreLabel}</span>
        <strong style={{ color: hasScore ? scoreColor(score) : 'var(--text-muted)' }}>{hasScore ? score.toFixed(2) : 'N/A'}</strong>
      </div>
      <div className="score-track">
        <div className="score-fill" style={{ width: hasScore ? `${Math.max(2, score * 100)}%` : '0%', background: hasScore ? scoreColor(score) : 'transparent' }} />
      </div>
      {footnote && <div className="muted" style={{ fontSize: '11.5px' }}>{footnote}</div>}
    </div>
  );
}

function AgentBreakdown({ breakdown, isPartial }) {
  const visual = breakdown?.visual_agent || {};
  const temporal = breakdown?.temporal_agent || {};
  const audio = breakdown?.audio_agent || {};
  const llm = breakdown?.llm_agent || {};
  const provenance = breakdown?.provenance_agent || {};
  // backend skips the face-swap model when signed C2PA credentials already prove AI generation
  const visualSkippedForC2pa = visual.status === 'skipped' && provenance.c2pa_ai_generated && provenance.c2pa_signature_valid;

  // llm_agent.reasoning starts with "<provider> ...analyzed N frames (Tools: ...)", then Gemini's summary after "): "
  const llmReasoning = llm.reasoning || '';
  const llmProvider = llmReasoning.startsWith('Gemini') ? 'Gemini' : llmReasoning.startsWith('Groq') ? 'Groq' : 'LLM';
  const summaryStart = llmReasoning.indexOf('): ');
  const llmSummary = llm.status === 'success' && summaryStart !== -1 ? llmReasoning.slice(summaryStart + 3) : llmReasoning;

  const ran = (agent) => agent.status === 'success' || agent.status === 'fallback';

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
      {isPartial && (
        <div className="banner banner-warning">
          <TriangleAlert size={18} />
          <div><strong>Partial analysis.</strong> Some agents failed, so their weight was shared among the agents that ran.</div>
        </div>
      )}

      {provenance.c2pa_present && (
        <div className={`banner ${provenance.c2pa_ai_generated ? 'banner-danger' : 'banner-success'}`}>
          {provenance.c2pa_ai_generated ? <ShieldAlert size={18} /> : <BadgeCheck size={18} />}
          <div>
            <strong>Content Credentials (C2PA) found.</strong>{' '}
            {provenance.c2pa_ai_generated
              ? `The file's signed metadata declares it AI-generated${provenance.c2pa_generator ? ` (signed with ${provenance.c2pa_generator})` : ''}. This alone is strong evidence, so the verdict is at least 95% manipulated.`
              : `Signed by ${provenance.c2pa_generator || 'an unknown tool'}, with no AI generation declared.`}
            {!provenance.c2pa_signature_valid && ' The signature could not be verified.'}
          </div>
        </div>
      )}

      <div className="section-title" style={{ marginBottom: 0, marginTop: '4px' }}>
        <Layers size={14} /> Agent evidence <span className="muted" style={{ textTransform: 'none', letterSpacing: 0, fontWeight: 500 }}>· score 0 = looks real, 1 = looks fake</span>
      </div>

      <div className="agent-grid">
        <AgentCard
          icon={ScanFace}
          accent="#7c83ff"
          title="Visual forensics"
          status={visual.status}
          badgeText={visual.status === 'skipped' ? (visualSkippedForC2pa ? 'Not needed' : 'No face') : undefined}
          description={visualSkippedForC2pa
            ? 'Skipped to save time: the signed C2PA credentials already prove this video is AI-generated.'
            : 'Face-swap detector (ViT) on frames that contain a face. It is not built to spot fully AI-generated video.'}
          scoreLabel="Face-swap score"
          score={ran(visual) ? visual.score || 0 : undefined}
        />
        <AgentCard
          icon={Activity}
          accent="#fbbf24"
          title="Temporal consistency"
          status={temporal.status}
          description="Compares consecutive frames for sudden motion spikes and facial-landmark jitter."
          scoreLabel="Anomaly score"
          score={ran(temporal) ? temporal.score || 0 : undefined}
          footnote={ran(temporal) && (temporal.score || 0) < 0.05 ? 'Smooth, natural motion. Modern AI video is often smooth too.' : null}
        />
        <AgentCard
          icon={AudioWaveform}
          accent="#f472b6"
          title="Audio & lip-sync"
          status={audio.has_audio ? audio.status : 'skipped'}
          badgeText={!audio.has_audio ? 'Silent' : undefined}
          description={audio.has_audio ? (audio.details?.summary || 'Spectral cutoff, voice cadence and mouth-to-voice sync.') : 'No audio track in this video.'}
          scoreLabel="Audio fake score"
          score={audio.has_audio && ran(audio) ? audio.score || 0 : undefined}
        />
        <AgentCard
          icon={BrainCircuit}
          accent="#22d3ee"
          title={`LLM reasoning · ${llmProvider}`}
          status={llm.status}
          description={llmSummary || 'A vision LLM reviews the most suspicious frames and explains what looks wrong.'}
          scoreLabel="Fake score"
          score={llm.status === 'success' ? llm.score || 0 : undefined}
        />
      </div>
    </div>
  );
}

export default AgentBreakdown;
