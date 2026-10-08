import React, { useEffect, useState } from 'react';
import { useParams, Link } from 'react-router-dom';
import { ArrowLeft, Download } from 'lucide-react';
import ThresholdMeter from './ThresholdMeter';
import { analysisService } from '../api';
import AgentBreakdown from './AgentBreakdown';
import FrameGallery from './FrameGallery';

const VERDICTS = {
  AUTHENTIC: { label: 'Authentic', color: 'var(--ok)', soft: 'var(--ok-soft)',
    summary: 'No agent found convincing evidence of manipulation.' },
  MANIPULATED: { label: 'Manipulated', color: 'var(--bad)', soft: 'var(--bad-soft)',
    summary: 'The agents found strong evidence that this video is AI-generated or altered.' },
  UNCERTAIN: { label: 'Uncertain', color: 'var(--warn)', soft: 'var(--warn-soft)',
    summary: 'The evidence is mixed or incomplete, so VeriFrame does not call it either way.' },
};

// time-per-stage bar: frame extraction plus each LangGraph node
const STAGE_META = {
  extract: { label: 'Frame extraction', color: '#b9b4a6' },
  cv_parallel: { label: 'CV agents', color: '#4a4e55' },
  llm: { label: 'LLM', color: '#17191c' },
  reflection: { label: 'Reflection', color: '#7c8088' },
  synthesis: { label: 'Synthesis', color: '#c2410c' },
  router: { label: 'Router', color: '#d8d3c6' },
  generative: { label: 'AI-gen detector', color: '#a15c07' },
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
      <div className="card" style={{ maxWidth: '560px', margin: '48px auto 0', textAlign: 'center', padding: '36px' }}>
        <h2 style={{ color: 'var(--danger)', fontSize: '18px', marginBottom: '10px' }}>Could not load the report</h2>
        <p style={{ color: 'var(--text-secondary)', marginBottom: '20px' }}>{error}</p>
        <Link to="/" className="btn btn-secondary"><ArrowLeft size={15} /> New analysis</Link>
      </div>
    );
  }

  const report = data.report || {};
  const thumbnails = data.thumbnails || [];
  const metadata = report.video_metadata || {};
  const performance = report.performance || {};
  const robustnessScore = metadata.robustness_score || 1.0;
  const verdict = VERDICTS[data.final_verdict] || VERDICTS.UNCERTAIN;
  const scorePct = Math.round((data.confidence || 0) * 100);

  const robustness = robustnessScore >= 0.8 ? ['High', 'var(--success)'] : robustnessScore >= 0.5 ? ['Medium', 'var(--warning)'] : ['Low', 'var(--danger)'];

  const stages = [
    ...(performance.extract_seconds != null ? [['extract', performance.extract_seconds]] : []),
    ...Object.entries(performance.stage_seconds || {}),
  ].filter(([, secs]) => secs >= 0.05);
  const stageTotal = stages.reduce((sum, [, secs]) => sum + secs, 0);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '22px' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: '12px', flexWrap: 'wrap' }}>
        <Link to="/" className="btn btn-ghost" style={{ padding: '8px 10px', fontSize: '13px', marginLeft: '-10px' }}>
          <ArrowLeft size={15} /> New analysis
        </Link>
        <button onClick={handleDownload} className="btn btn-secondary" disabled={downloading} style={{ padding: '8px 14px', fontSize: '13px' }}>
          <Download size={15} /> {downloading ? 'Preparing…' : 'Export report'}
        </button>
      </div>

      <div className="card verdict" style={{ '--verdict-color': verdict.color, '--verdict-soft': verdict.soft }}>
        <div className="verdict-main">
          <span className="label">Verdict · case {jobId.slice(0, 8)}</span>
          <div className="verdict-word" style={{ marginTop: '8px' }}>{verdict.label}</div>
          <p style={{ color: 'var(--ink-2)', marginTop: '10px', maxWidth: '56ch' }}>{verdict.summary}</p>
          <div className="file-name">{data.video_filename}</div>
          <div className="facts" style={{ marginTop: '14px' }}>
            <div className="fact"><span className="label">Scanned</span><strong>{new Date(data.created_at).toLocaleDateString()}</strong></div>
            <div className="fact"><span className="label">Duration</span><strong>{data.duration.toFixed(1)} s</strong></div>
            {metadata.width && <div className="fact"><span className="label">Resolution</span><strong>{metadata.width}×{metadata.height}</strong></div>}
            <div className="fact"><span className="label">Robustness</span><strong style={{ color: robustness[1] }}>{robustnessScore.toFixed(2)} {robustness[0]}</strong></div>
            {performance.total_seconds != null && <div className="fact"><span className="label">Analyzed in</span><strong>{performance.total_seconds.toFixed(1)} s</strong></div>}
          </div>
        </div>
        <div className="verdict-side">
          <div>
            <span className="label">Manipulation score</span>
            <div className="verdict-score" style={{ marginTop: '8px' }}>{scorePct}<small>/ 100</small></div>
          </div>
          <ThresholdMeter score={data.confidence || 0} color={verdict.color} />
        </div>
      </div>

      {stages.length > 0 && (
        <div className="card">
          <div className="section-title">
            <h2>Time per stage</h2>
            {performance.llm_calls > 0 && (
              <span className="label">
                {performance.llm_calls} LLM call{performance.llm_calls > 1 ? 's' : ''} · {((performance.llm_input_tokens || 0) / 1000).toFixed(1)}K tokens
                {performance.llm_cost_usd != null && ` · $${performance.llm_cost_usd.toFixed(4)}`}
              </span>
            )}
          </div>
          <div className="timing-bar">
            {stages.map(([key, secs]) => (
              <span key={key} title={`${STAGE_META[key]?.label || key}: ${secs.toFixed(1)}s`}
                style={{ flex: secs / stageTotal, background: STAGE_META[key]?.color || '#7c8088' }} />
            ))}
          </div>
          <div className="timing-legend">
            {stages.map(([key, secs]) => (
              <span key={key}>
                <i style={{ background: STAGE_META[key]?.color || '#7c8088' }} />
                {STAGE_META[key]?.label || key} <span className="mono" style={{ color: 'var(--ink)' }}>{secs.toFixed(1)}s</span>
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
