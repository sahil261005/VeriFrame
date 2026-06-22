import React, { useEffect, useState } from 'react';
import { useParams, Link } from 'react-router-dom';
import { analysisService } from '../api';
import AgentBreakdown from './AgentBreakdown';
import ConfidenceChart from './ConfidenceChart';
import FrameGallery from './FrameGallery';

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
        <div className="spinner" style={{ width: '30px', height: '30px' }} />
      </div>
    );
  }

  if (error) {
    return (
      <div style={{ maxWidth: '600px', margin: '40px auto' }}>
        <div className="card" style={{ textAlign: 'center', padding: '40px' }}>
          <h2 style={{ color: 'var(--danger)', fontSize: '18px', marginBottom: '10px' }}>Failed to Load</h2>
          <p style={{ color: 'var(--text-secondary)', marginBottom: '20px' }}>{error}</p>
          <Link to="/" className="btn btn-secondary">
            ← Scan New Video
          </Link>
        </div>
      </div>
    );
  }

  const report = data.report || {};
  const thumbnails = data.thumbnails || [];
  const confidencePercent = (data.confidence * 100).toFixed(0);

  // verdict selection badge helper
  const getVerdictBadge = () => {
    switch (data.final_verdict) {
      case 'AUTHENTIC':
        return <span className="badge badge-authentic">Authentic</span>;
      case 'MANIPULATED':
        return <span className="badge badge-manipulated">Manipulated</span>;
      default:
        return <span className="badge badge-uncertain">Uncertain</span>;
    }
  };

  const getVerdictColor = () => {
    switch (data.final_verdict) {
      case 'AUTHENTIC': return 'var(--success)';
      case 'MANIPULATED': return 'var(--danger)';
      default: return 'var(--warning)';
    }
  };

  return (
    <div style={{ width: '100%' }}>
      {/* back navigation */}
      <div style={{ marginBottom: '20px', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <Link to="/" style={{ fontSize: '13px', fontWeight: '500', color: 'var(--primary)', textDecoration: 'none', display: 'flex', alignItems: 'center', gap: '4px' }}>
          ← Scan new video
        </Link>
        
        <button onClick={handleDownload} className="btn" disabled={downloading} style={{ padding: '8px 16px', fontSize: '13px' }}>
          {downloading ? 'Downloading...' : 'Export Report (PDF)'}
        </button>
      </div>

      {/* header overview card */}
      <div className="card" style={{ marginBottom: '20px' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '20px' }}>
          <div>
            <div style={{ display: 'flex', gap: '8px', alignItems: 'center', marginBottom: '6px' }}>
              <h2 style={{ fontSize: '16px', fontWeight: '700', color: 'var(--text-primary)', wordBreak: 'break-all' }}>
                {data.video_filename}
              </h2>
            </div>
            <div style={{ display: 'flex', gap: '16px', fontSize: '12px', color: 'var(--text-muted)' }}>
              <span>
                Scanned: {new Date(data.created_at).toLocaleDateString()}
              </span>
              <span>
                Duration: {data.duration.toFixed(2)}s
              </span>
            </div>
          </div>
          
          <div style={{ display: 'flex', gap: '24px', alignItems: 'center', flexWrap: 'wrap' }}>
            <div style={{ textAlign: 'right' }}>
              <div style={{ fontSize: '11px', color: 'var(--text-muted)', textTransform: 'uppercase', marginBottom: '4px' }}>
                Final Verdict
              </div>
              <div>{getVerdictBadge()}</div>
            </div>

            {/* simple confidence percentage card block */}
            <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
              <div style={{ 
                backgroundColor: 'var(--bg-primary)', 
                border: '1px solid var(--border-color)', 
                borderRadius: '4px',
                padding: '6px 12px',
                fontSize: '20px',
                fontWeight: '700',
                color: getVerdictColor() 
              }}>
                {confidencePercent}%
              </div>
              <div style={{ maxWidth: '120px' }}>
                <div style={{ fontSize: '13px', fontWeight: '600', color: 'var(--text-primary)' }}>Consensus Score</div>
                <div style={{ fontSize: '11px', color: 'var(--text-muted)' }}>Pipeline confidence rating</div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* details grid layout */}
      <div className="dashboard-grid">
        <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
          <AgentBreakdown breakdown={report.agent_breakdown} isPartial={data.is_partial_analysis} />
          <FrameGallery thumbnails={thumbnails} explanations={report.frame_level_details} />
        </div>
        
        <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
          <ConfidenceChart report={report} duration={data.duration} />
        </div>
      </div>
    </div>
  );
}

export default ResultsDashboard;

