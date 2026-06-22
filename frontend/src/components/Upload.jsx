import React, { useState, useRef } from 'react';
import { analysisService } from '../api';

function Upload({ onUploadSuccess }) {
  const [dragActive, setDragActive] = useState(false);
  const [file, setFile] = useState(null);
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const fileInputRef = useRef(null);

  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const droppedFile = e.dataTransfer.files[0];
      validateAndSetFile(droppedFile);
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
    
    // validate file extension format
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
    <div style={{ maxWidth: '600px', width: '100%', margin: '40px auto 0 auto' }}>
      <div className="card" style={{ padding: '40px' }}>
        <h2 style={{ fontSize: '22px', fontWeight: '700', marginBottom: '8px', textAlign: 'center' }}>
          Scan New Video
        </h2>
        <p style={{ fontSize: '14px', color: 'var(--text-secondary)', textAlign: 'center', marginBottom: '30px' }}>
          Upload a media clip to test for deepfake anomalies and visual/temporal manipulations.
        </p>

        {error && (
          <div style={{ padding: '12px', borderRadius: '6px', backgroundColor: 'rgba(239,68,68,0.1)', border: '1px solid rgba(239,68,68,0.2)', color: '#ef4444', fontSize: '14px', marginBottom: '20px', display: 'flex', gap: '8px', alignItems: 'center' }}>
            <span>{error}</span>
          </div>
        )}

        <form onDragEnter={handleDrag} onSubmit={(e) => e.preventDefault()} style={{ width: '100%' }}>
          <input
            ref={fileInputRef}
            type="file"
            className="form-input"
            style={{ display: 'none' }}
            accept=".mp4,.avi,.mov,.webm"
            onChange={handleChange}
          />

          <div
            onClick={() => fileInputRef.current.click()}
            onDragEnter={handleDrag}
            onDragOver={handleDrag}
            onDragLeave={handleDrag}
            onDrop={handleDrop}
            style={{
              border: '1px solid var(--border-color)',
              borderRadius: '6px',
              padding: '40px 20px',
              textAlign: 'center',
              cursor: 'pointer',
              backgroundColor: dragActive ? 'var(--primary-light)' : '#f8fafc',
              borderColor: dragActive ? 'var(--primary)' : 'var(--border-color)',
              transition: 'var(--transition)'
            }}
          >
            {file ? (
              <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '12px' }}>
                <div>
                  <div style={{ fontWeight: '600', fontSize: '15px', color: 'var(--text-primary)', maxWidth: '300px', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                    Selected File: {file.name}
                  </div>
                  <div style={{ fontSize: '13px', color: 'var(--text-secondary)', marginTop: '4px' }}>
                    {(file.size / (1024 * 1024)).toFixed(2)} MB
                  </div>
                </div>
              </div>
            ) : (
              <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '12px' }}>
                <div style={{ color: 'var(--text-secondary)' }}>
                  <span style={{ color: 'var(--primary)', fontWeight: '600' }}>Click to select video</span> or drag and drop here
                </div>
                <div style={{ fontSize: '12px', color: 'var(--text-muted)', marginTop: '4px' }}>
                  Supports MP4, AVI, MOV, WEBM (Max duration: 30s)
                </div>
              </div>
            )}
          </div>

          {file && (
            <div style={{ display: 'flex', gap: '15px', marginTop: '24px' }}>
              <button
                type="button"
                className="btn btn-secondary"
                style={{ flex: 1 }}
                onClick={() => setFile(null)}
                disabled={loading}
              >
                Clear Selection
              </button>
              <button
                type="button"
                className="btn"
                style={{ flex: 2 }}
                onClick={handleUpload}
                disabled={loading}
              >
                {loading ? 'Uploading...' : 'Scan Video'}
              </button>
            </div>
          )}
        </form>
      </div>
    </div>
  );
}

export default Upload;

