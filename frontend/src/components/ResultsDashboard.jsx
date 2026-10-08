import React, { useEffect, useState } from 'react';
import { useParams, Link } from 'react-router-dom';
import { ArrowLeft, Download, Clock, Film, Maximize, ShieldCheck, ShieldAlert, ShieldQuestion, Timer, Gauge } from 'lucide-react';
import { analysisService } from '../api';
import AgentBreakdown from './AgentBreakdown';
import FrameGallery from './FrameGallery';

const VERDICTS = {
  AUTHENTIC: { label: 'Authentic', badge: 'badge-authentic', color: 'var(--success)', glow: 'rgba(52, 211, 153, 0.22)', icon: ShieldCheck,
    summary: 'No agent found convincing evidence of manipulation.' },
  MANIPULATED: { label: 'Manipulated', badge: 'badge-manipulated', color: 'var(--danger)', glow: 'rgba(248, 113, 113, 0.22)', icon: ShieldAlert,
    summary: 'The agents found strong evidence that this video is AI-generated or altered.' },
  UNCERTAIN: { label: 'Uncertain', badge: 'badge-uncertain', color: 'var(--warning)', glow: 'rgba(251, 191, 36, 0.2)', icon: ShieldQuestion,
    summary: 'The evidence is mixed or incomplete, so VeriFrame will not call it either way.' },
};

// time-per-stage bar: frame extraction plus each LangGraph node
const STAGE_META = {
  extract: { label: 'Frame extraction', color: '#64748b' },
  cv_parallel: { label: 'CV agents', color: '#7c83ff' },
  llm: { label: 'LLM', color: '#22d3ee' },
  reflection: { label: 'Reflection', color: '#a78bfa' },
  synthesis: { label: 'Synthesis', color: '#34d399' },
  router: { label: 'Router', color: '#fbbf24' },
  generative: { label: 'AI-gen detector', color: '#f472b6' },
};

function ResultsDashboard() {
  const { jobId } = useParams();
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [downloading, setDownloading] = useState(false);

  useEffect(() => {
    const fetchResults = async () => {
      try {
        const result = await analysisService.getAnalysis(jobId);
        setData(result);
      } catch (err) {
        setError(err.response?.data?.detail || 'Failed to load analysis report.');
      } finally {
        setLoading(false);
      }
    };
    fetchResults();
  }, [jobId]);

  const handleDownload = async () => {
    if (downloading) return;
    setDownloading(true);
    try {
      const pdfBlob = await analysisService.getPDFReport(jobId);
      const url = window.URL.createObjectURL(pdfBlob);
      const a = document.createElement('a');
      a.href = url;
      const isHtml = pdfBlob.type === 'text/html';
      a.download = `veriframe_report_${jobId}.${isHtml ? 'html' : 'pdf'}`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      window.URL.revokeObjectURL(url);
    } catch (err) {
      console.error('Error downloading report:', err);
      alert('Could not download PDF report.');
    } finally {
      setDownloading(false);
    }
  };

  if (loading) {
    return (
      <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '60vh' }}>
        <div className="spinner" style={{ width: '32px', height: '32px' }} />
      </div>
    );
  }

  if (error) {
    return (
      <div className="card fade-up" style={{ maxWidth: '560px', margin: '48px auto 0', textAlign: 'center', padding: '36px' }}>
        <h2 style={{ color: 'var(--danger)', fontSize: '18px', marginBottom: '10px' }}>Could not load the report</h2>
        <p style={{ color: 'var(--text-secondary)', marginBottom: '20px' }}>{error}</p>
        <Link to="/" className="btn btn-secondary"><ArrowLeft size={15} /> Scan a new video</Link>
      </div>
    );
  }

  const report = data.report || {};
  const thumbnails = data.thumbnails || [];
  const metadata = report.video_metadata || {};
  const performance = report.performance || {};
  const robustnessScore = metadata.robustness_score || 1.0;
  const verdict = VERDICTS[data.final_verdict] || VERDICTS.UNCERTAIN;
  const VerdictIcon = verdict.icon;
  const scorePct = Math.round((data.confidence || 0) * 100);

  const robustness = robustnessScore >= 0.8 ? ['High', 'var(--success)'] : robustnessScore >= 0.5 ? ['Medium', 'var(--warning)'] : ['Low', 'var(--danger)'];

  const stages = [
    ...(performance.extract_seconds != null ? [['extract', performance.extract_seconds]] : []),
    ...Object.entries(performance.stage_seconds || {}),
  ].filter(([, secs]) => secs >= 0.05);
  const stageTotal = stages.reduce((sum, [, secs]) => sum + secs, 0);

  const size = 168;
  const stroke = 12;
  const radius = (size - stroke) / 2;
  const circumference = 2 * Math.PI * radius;

  return (
    <div className="fade-up" style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: '12px', flexWrap: 'wrap' }}>
        <Link to="/" className="btn btn-ghost" style={{ padding: '8px 10px', fontSize: '13px' }}>
          <ArrowLeft size={15} /> Scan a new video
        </Link>
        <button onClick={handleDownload} className="btn btn-secondary" disabled={downloading} style={{ padding: '8px 14px', fontSize: '13px' }}>
          <Download size={15} /> {downloading ? 'Preparing…' : 'Export report'}
        </button>
      </div>

      {/* verdict hero */}
      <div className="card verdict-hero" style={{ '--verdict-glow': verdict.glow, padding: '28px' }}>
        <div style={{ minWidth: 0 }}>
          <span className={`badge ${verdict.badge}`} style={{ fontSize: '13px', padding: '6px 12px', marginBottom: '14px' }}>
            <VerdictIcon size={15} /> {verdict.label}
          </span>
          <h1 className="verdict-title">{data.video_filename}</h1>
          <p style={{ color: 'var(--text-secondary)', fontSize: '14.5px', marginBottom: '16px', maxWidth: '560px' }}>{verdict.summary}</p>
          <div className="chip-row">
            <span className="chip"><Clock size={13} /> {new Date(data.created_at).toLocaleDateString()}</span>
            <span className="chip"><Film size={13} /> <strong>{data.duration.toFixed(1)}s</strong></span>
            {metadata.width && <span className="chip"><Maximize size={13} /> <strong>{metadata.width}×{metadata.height}</strong></span>}
            <span className="chip"><Gauge size={13} /> Robustness <strong style={{ color: robustness[1] }}>{robustnessScore.toFixed(2)} {robustness[0]}</strong></span>
            {performance.total_seconds != null && (
              <span className="chip"><Timer size={13} /> Analyzed in <strong>{performance.total_seconds.toFixed(1)}s</strong></span>
            )}
          </div>
        </div>

        <div className="gauge">
          <svg width={size} height={size}>
            <circle cx={size / 2} cy={size / 2} r={radius} fill="none" stroke="rgba(148,163,184,0.12)" strokeWidth={stroke} />
            <circle
              cx={size / 2}
              cy={size / 2}
              r={radius}
              fill="none"
              stroke={verdict.color}
              strokeWidth={stroke}
              strokeLinecap="round"
              strokeDasharray={circumference}
              strokeDashoffset={circumference * (1 - scorePct / 100)}
              style={{ transition: 'stroke-dashoffset 1s ease', filter: `drop-shadow(0 0 8px ${verdict.glow})` }}
            />
          </svg>
          <div className="gauge-center">
            <div className="gauge-value" style={{ color: verdict.color }}>{scorePct}%</div>
            <div className="gauge-caption">manipulation score</div>
          </div>
        </div>
      </div>

      {/* time per stage */}
      {stages.length > 0 && (
        <div className="card">
          <div className="section-title" style={{ marginBottom: '4px', justifyContent: 'space-between' }}>
            <span style={{ display: 'flex', alignItems: 'center', gap: '8px' }}><Timer size={14} /> Time per stage</span>
            {performance.llm_calls > 0 && (
              <span className="mono" style={{ textTransform: 'none', letterSpacing: 0, fontWeight: 500 }}>
                {performance.llm_calls} LLM call{performance.llm_calls > 1 ? 's' : ''} · {((performance.llm_input_tokens || 0) / 1000).toFixed(1)}K tokens
                {performance.llm_cost_usd != null && ` · $${performance.llm_cost_usd.toFixed(4)}`}
              </span>
            )}
          </div>
          <div className="timing-bar">
            {stages.map(([key, secs]) => (
              <span key={key} title={`${STAGE_META[key]?.label || key}: ${secs.toFixed(1)}s`}
                style={{ width: `${(secs / stageTotal) * 100}%`, background: STAGE_META[key]?.color || '#94a3b8' }} />
            ))}
          </div>
          <div className="timing-legend">
            {stages.map(([key, secs]) => (
              <span key={key}>
                <i style={{ background: STAGE_META[key]?.color || '#94a3b8' }} />
                {STAGE_META[key]?.label || key} <span className="mono" style={{ color: 'var(--text-primary)' }}>{secs.toFixed(1)}s</span>
              </span>
            ))}
          </div>
        </div>
      )}

      <AgentBreakdown breakdown={report.agent_breakdown} isPartial={data.is_partial_analysis} />
      <FrameGallery thumbnails={thumbnails} explanations={report.frame_level_details} />
    </div>
  );
}

export default ResultsDashboard;
