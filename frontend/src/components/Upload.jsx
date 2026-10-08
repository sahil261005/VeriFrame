import React, { useState, useRef } from 'react';
import { CloudUpload, FileVideo, ScanFace, Activity, AudioWaveform, BadgeCheck, BrainCircuit, Scale, TriangleAlert, X } from 'lucide-react';
import { analysisService } from '../api';

const AGENTS = [
  { icon: ScanFace, name: 'Visual', desc: 'Face-swap ViT' },
  { icon: Activity, name: 'Temporal', desc: 'Motion & landmarks' },
  { icon: AudioWaveform, name: 'Audio', desc: 'Voice & lip-sync' },
  { icon: BadgeCheck, name: 'Provenance', desc: 'C2PA credentials' },
  { icon: BrainCircuit, name: 'LLM', desc: 'Gemini · Groq fallback' },
  { icon: Scale, name: 'Consensus', desc: 'Weighted verdict' },
];

function Upload({ onUploadSuccess }) {
  const [dragActive, setDragActive] = useState(false);
  const [file, setFile] = useState(null);
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const fileInputRef = useRef(null);

  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      validateAndSetFile(e.dataTransfer.files[0]);
    }
  };

  const handleChange = (e) => {
    e.preventDefault();
    if (e.target.files && e.target.files[0]) {
      validateAndSetFile(e.target.files[0]);
    }
  };

  const validateAndSetFile = (selectedFile) => {
    setError('');
    const ext = selectedFile.name.split('.').pop().toLowerCase();
    const allowed = ['mp4', 'avi', 'mov', 'webm'];
    if (!allowed.includes(ext)) {
      setError(`Unsupported video format .${ext}. Only MP4, AVI, MOV, and WEBM are supported.`);
      setFile(null);
      return;
    }
    setFile(selectedFile);
  };

  const handleUpload = async () => {
    if (!file) return;
    setError('');
    setLoading(true);
    try {
      const data = await analysisService.uploadVideo(file);
      if (data && data.id) {
        onUploadSuccess(data.id);
      }
    } catch (err) {
      setError(err.response?.data?.detail || 'An error occurred during video upload.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="fade-up">
      <div className="hero">
        <span className="eyebrow"><span className="live-dot" /> 6 forensic agents · LangGraph orchestration</span>
        <h1>Is this video <span className="gradient-text">real?</span></h1>
        <p>Upload a clip. Specialist agents check faces, motion, audio and provenance in parallel, and an LLM explains what it sees, frame by frame.</p>
      </div>

      <div className="card" style={{ maxWidth: '760px', margin: '0 auto', padding: '26px' }}>
        {error && (
          <div className="banner banner-danger" style={{ marginBottom: '18px' }}>
            <TriangleAlert size={18} />
            <span>{error}</span>
          </div>
        )}

        <form onDragEnter={handleDrag} onSubmit={(e) => e.preventDefault()}>
          <input
            ref={fileInputRef}
            type="file"
            style={{ display: 'none' }}
            accept=".mp4,.avi,.mov,.webm"
            onChange={handleChange}
          />

          {file ? (
            <div className="dropzone" style={{ cursor: 'default', padding: '28px 20px' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '14px', justifyContent: 'center', textAlign: 'left' }}>
                <div className="dropzone-icon" style={{ margin: 0 }}><FileVideo size={26} /></div>
                <div style={{ minWidth: 0 }}>
                  <div style={{ fontWeight: 600, fontSize: '15px', maxWidth: '420px', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                    {file.name}
                  </div>
                  <div className="mono muted" style={{ fontSize: '12.5px', marginTop: '2px' }}>
                    {(file.size / (1024 * 1024)).toFixed(2)} MB · ready to scan
                  </div>
                </div>
                <button type="button" className="btn btn-ghost" onClick={() => setFile(null)} disabled={loading} aria-label="Clear selection" style={{ padding: '8px' }}>
                  <X size={18} />
                </button>
              </div>
            </div>
          ) : (
            <div
              className={`dropzone ${dragActive ? 'active' : ''}`}
              onClick={() => fileInputRef.current.click()}
              onDragEnter={handleDrag}
              onDragOver={handleDrag}
              onDragLeave={handleDrag}
              onDrop={handleDrop}
            >
              <div className="dropzone-icon"><CloudUpload size={26} /></div>
              <div style={{ fontSize: '15px', color: 'var(--text-secondary)' }}>
                <span style={{ color: 'var(--text-primary)', fontWeight: 600 }}>Click to choose a video</span> or drag and drop it here
              </div>
              <div className="muted" style={{ fontSize: '12.5px', marginTop: '6px' }}>
                MP4, MOV, AVI or WEBM · up to 30 seconds · up to ~720p
              </div>
            </div>
          )}

          {file && (
            <button type="button" className="btn" style={{ width: '100%', marginTop: '16px', padding: '13px' }} onClick={handleUpload} disabled={loading}>
              {loading ? <><span className="spinner" style={{ width: 16, height: 16 }} /> Uploading…</> : <><ScanFace size={17} /> Run forensic analysis</>}
            </button>
          )}
        </form>

        <div className="agent-strip">
          {AGENTS.map(({ icon: Icon, name, desc }) => (
            <div key={name} className="agent-strip-item">
              <Icon size={18} />
              <div><strong>{name}</strong>{desc}</div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

export default Upload;
