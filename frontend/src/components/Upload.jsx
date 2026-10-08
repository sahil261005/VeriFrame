import React, { useState, useRef } from 'react';
import { CloudUpload, FileVideo, ScanFace, Activity, AudioWaveform, BadgeCheck, BrainCircuit, Scale, TriangleAlert, X } from 'lucide-react';
import { analysisService } from '../api';

const MAX_UPLOAD_MB = 50; // keep in step with MAX_UPLOAD_MB on the backend

const CHECKS = [
  { icon: ScanFace, name: 'Face-swap detector', desc: 'Vision Transformer on each keyframe', tech: 'ViT · ONNX' },
  { icon: Activity, name: 'Motion consistency', desc: 'Flicker and landmark jitter between frames', tech: 'optical flow' },
  { icon: AudioWaveform, name: 'Audio and lip-sync', desc: 'Synthetic-voice cues, mouth-to-voice timing', tech: 'spectral' },
  { icon: BadgeCheck, name: 'Content Credentials', desc: 'Signed "made by AI" labels inside the file', tech: 'C2PA' },
  { icon: BrainCircuit, name: 'Vision LLM', desc: 'Explains what looks wrong in the riskiest frames', tech: 'Gemini · Groq' },
  { icon: Scale, name: 'Consensus', desc: 'Weighs the evidence; says Uncertain when it is thin', tech: 'LangGraph' },
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
    if (selectedFile.size > MAX_UPLOAD_MB * 1024 * 1024) {
      setError(`Video is too large (${(selectedFile.size / (1024 * 1024)).toFixed(0)} MB). The limit is ${MAX_UPLOAD_MB} MB.`);
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
    <div className="intake">
      <div>
        <span className="label">New case</span>
        <h1>Check a video before you <em>trust</em> it.</h1>
        <p className="intake-lede">
          Six forensic checks run on your clip. Each one reports its own evidence, and VeriFrame
          explains its verdict frame by frame.
        </p>
        <div className="checks">
          {CHECKS.map(({ icon: Icon, name, desc, tech }) => (
            <div key={name} className="check-row">
              <Icon size={16} />
              <div><strong>{name}</strong> <span className="muted">· {desc}</span></div>
              <span className="mono">{tech}</span>
            </div>
          ))}
        </div>
      </div>

      <div className="card" style={{ padding: '22px' }}>
        <div className="section-title"><h2>Upload evidence</h2><span className="label">mp4 · mov · avi · webm</span></div>

        {error && (
          <div className="banner banner-danger" style={{ marginBottom: '16px' }}>
            <TriangleAlert size={18} />
            <span>{error}</span>
          </div>
        )}

        <form onDragEnter={handleDrag} onSubmit={(e) => e.preventDefault()}>
          <input
            id="video-file"
            ref={fileInputRef}
            type="file"
            style={{ display: 'none' }}
            accept=".mp4,.avi,.mov,.webm"
            onChange={handleChange}
          />

          {file ? (
            <div className="dropzone" style={{ cursor: 'default', padding: '22px', textAlign: 'left' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '14px' }}>
                <div className="dropzone-icon" style={{ margin: 0, flexShrink: 0 }}><FileVideo size={22} /></div>
                <div style={{ minWidth: 0, flex: 1 }}>
                  <div className="mono" style={{ fontSize: '14px', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{file.name}</div>
                  <div className="muted" style={{ fontSize: '13px' }}>{(file.size / (1024 * 1024)).toFixed(2)} MB</div>
                </div>
                <button type="button" className="btn btn-ghost" onClick={() => setFile(null)} disabled={loading} aria-label="Remove file" style={{ padding: '8px' }}>
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
              <div className="dropzone-icon"><CloudUpload size={22} /></div>
              <div style={{ fontSize: '15px' }}>
                <strong>Choose a video</strong> <span className="muted">or drop it here</span>
              </div>
              <div className="muted" style={{ fontSize: '13px', marginTop: '6px' }}>Up to 30 seconds and 1080p</div>
            </div>
          )}

          <button type="button" className="btn" style={{ width: '100%', marginTop: '14px', padding: '12px' }} onClick={handleUpload} disabled={!file || loading}>
            {loading ? <><span className="spinner" style={{ width: 15, height: 15, borderColor: 'rgba(255,255,255,0.3)', borderLeftColor: '#fff' }} /> Uploading…</> : 'Analyze video'}
          </button>
        </form>
      </div>
    </div>
  );
}

export default Upload;
