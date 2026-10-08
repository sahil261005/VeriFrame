import React, { useState, useEffect } from 'react';
import { createPortal } from 'react-dom';
import { Eye, X, BrainCircuit } from 'lucide-react';

function FrameGallery({ thumbnails, explanations }) {
  const [selectedFrame, setSelectedFrame] = useState(null);

  useEffect(() => {
    if (!selectedFrame) return;
    const onKey = (e) => e.key === 'Escape' && setSelectedFrame(null);
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [selectedFrame]);

  const frames = (thumbnails || []).map((item) => {
    const tsVal = Number(item.timestamp);
    // keys may arrive as "1.333" or "1.333s"; match within 0.15s to absorb rounding
    const matchedKey = Object.keys(explanations || {}).find((key) => Math.abs(parseFloat(key) - tsVal) < 0.15);
    const explanation = matchedKey && explanations[matchedKey] ? explanations[matchedKey] : 'No LLM explanation is available for this frame.';
    return { ...item, explanation };
  });

  return (
    <div className="card">
      <div className="section-title" style={{ justifyContent: 'space-between' }}>
        <span style={{ display: 'flex', alignItems: 'center', gap: '8px' }}><Eye size={14} /> Frames reviewed by the LLM</span>
        {frames.length > 0 && <span className="muted" style={{ textTransform: 'none', letterSpacing: 0, fontWeight: 500 }}>Click a frame for the full explanation</span>}
      </div>

      {frames.length === 0 ? (
        <p className="muted" style={{ fontSize: '13px' }}>No frames were sent to the LLM.</p>
      ) : (
        <div className="frame-grid">
          {frames.map((frame, idx) => (
            <div key={idx} className="frame-card" onClick={() => setSelectedFrame(frame)}>
              <div className="frame-thumb">
                <img src={frame.image_b64} alt={`Frame at ${frame.timestamp}s`} />
                <span className="frame-ts">t={frame.timestamp}s</span>
              </div>
              <div className="frame-text"><span>{frame.explanation}</span></div>
            </div>
          ))}
        </div>
      )}

      {/* portal to <body>: the page's fade-in animation makes it the containing block for position:fixed */}
      {selectedFrame && createPortal(
        <div className="modal-backdrop" onClick={() => setSelectedFrame(null)}>
          <div className="card modal" onClick={(e) => e.stopPropagation()}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '14px' }}>
              <h3 style={{ fontSize: '16px', fontWeight: 700 }}>
                Frame at <span className="mono">{selectedFrame.timestamp}s</span>
              </h3>
              <button className="btn btn-ghost" onClick={() => setSelectedFrame(null)} aria-label="Close" style={{ padding: '6px' }}>
                <X size={18} />
              </button>
            </div>
            <img
              src={selectedFrame.image_b64}
              alt={`Frame at ${selectedFrame.timestamp}s`}
              style={{ width: '100%', maxHeight: '340px', objectFit: 'contain', borderRadius: '10px', background: '#000', border: '1px solid var(--border)', marginBottom: '14px' }}
            />
            <div className="section-title" style={{ marginBottom: '6px' }}><BrainCircuit size={14} /> LLM forensic note</div>
            <p style={{ fontSize: '14px', lineHeight: 1.6, color: 'var(--text-primary)' }}>{selectedFrame.explanation}</p>
          </div>
        </div>,
        document.body
      )}
    </div>
  );
}

export default FrameGallery;
