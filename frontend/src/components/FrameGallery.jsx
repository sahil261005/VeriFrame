import React, { useState } from 'react';

function FrameGallery({ thumbnails, explanations }) {
  const [selectedFrame, setSelectedFrame] = useState(null);

  if (!thumbnails || thumbnails.length === 0) {
    return (
      <div className="card" style={{ padding: '20px' }}>
        <h4 style={{ fontSize: '15px', fontWeight: '700', marginBottom: '10px' }}>Flagged Suspicious Frames</h4>
        <p style={{ color: 'var(--text-secondary)', fontSize: '13px' }}>No frames flagged for deepfake reasoning.</p>
      </div>
    );
  }

  return (
    <div className="card" style={{ padding: '20px' }}>
      <h4 style={{ fontSize: '15px', fontWeight: '700', marginBottom: '12px' }}>Flagged Suspicious Frames</h4>
      
      <div style={{
        display: 'grid',
        gridTemplateColumns: 'repeat(auto-fill, minmax(100px, 1fr))',
        gap: '12px'
      }}>
        {thumbnails.map((item, idx) => {
          const tsVal = Number(item.timestamp);
          const matchedKey = Object.keys(explanations || {}).find(key => Number(key) === tsVal);
          const explanation = matchedKey ? explanations[matchedKey] : "No detailed forensic analysis available for this frame.";

          return (
            <div 
              key={idx}
              onClick={() => setSelectedFrame({ ...item, explanation })}
              style={{
                position: 'relative',
                borderRadius: '4px',
                overflow: 'hidden',
                cursor: 'pointer',
                border: '1px solid var(--border-color)',
                backgroundColor: '#000',
                transition: 'var(--transition)'
              }}
              className="gallery-card"
            >
              {/* image */}
              <img 
                src={item.image_b64} 
                alt={`Flagged t=${item.timestamp}s`}
                style={{
                  width: '100%',
                  height: '70px',
                  objectFit: 'cover',
                  display: 'block'
                }}
              />
              
              {/* overlay hover */}
              <div style={{
                position: 'absolute',
                top: 0, left: 0, right: 0, bottom: 0,
                backgroundColor: 'rgba(0,0,0,0.4)',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                opacity: 0,
                transition: 'var(--transition)',
                color: '#fff',
                fontSize: '11px',
                fontWeight: 'bold'
              }}
              className="gallery-hover"
              >
                [ VIEW ]
              </div>

              {/* timestamp tag */}
              <div style={{
                position: 'absolute',
                bottom: '4px',
                left: '4px',
                backgroundColor: 'rgba(0,0,0,0.65)',
                padding: '2px 4px',
                borderRadius: '3px',
                fontSize: '9px',
                fontWeight: '600',
                color: '#fff'
              }}>
                t={item.timestamp}s
              </div>
            </div>
          );
        })}
      </div>

      {/* modal overlay for selected frame explanation */}
      {selectedFrame && (
        <div style={{
          position: 'fixed',
          top: 0, left: 0, right: 0, bottom: 0,
          backgroundColor: 'rgba(15, 23, 42, 0.65)',
          backdropFilter: 'blur(4px)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          zIndex: 1000,
          padding: '20px'
        }}>
          <div className="card" style={{
            maxWidth: '420px',
            width: '100%',
            position: 'relative',
            padding: '20px',
            boxShadow: '0 10px 25px rgba(0, 0, 0, 0.15)'
          }}>
            <button 
              onClick={() => setSelectedFrame(null)}
              style={{
                position: 'absolute',
                top: '12px',
                right: '12px',
                background: 'none',
                border: 'none',
                color: 'var(--text-secondary)',
                cursor: 'pointer',
                fontWeight: 'bold',
                fontSize: '14px'
              }}
            >
              ✕
            </button>

            <h3 style={{ fontSize: '15px', fontWeight: '700', marginBottom: '12px', color: 'var(--text-primary)' }}>
              Forensic Analysis (t={selectedFrame.timestamp}s)
            </h3>

            <img 
              src={selectedFrame.image_b64} 
              alt={`Flagged large t=${selectedFrame.timestamp}s`}
              style={{
                width: '100%',
                maxHeight: '200px',
                objectFit: 'contain',
                borderRadius: '4px',
                backgroundColor: '#000',
                marginBottom: '12px',
                border: '1px solid var(--border-color)'
              }}
            />

            <div>
              <div style={{ fontSize: '11px', color: 'var(--text-muted)', textTransform: 'uppercase', fontWeight: '700', letterSpacing: '0.05em', marginBottom: '4px' }}>
                AI Forensic Verdict
              </div>
              <p style={{ fontSize: '13px', lineHeight: '1.45', color: 'var(--text-primary)' }}>
                {selectedFrame.explanation}
              </p>
            </div>
          </div>
        </div>
      )}

      {/* hover styling injector */}
      <style>{`
        .gallery-card:hover .gallery-hover {
          opacity: 1 !important;
        }
      `}</style>
    </div>
  );
}

export default FrameGallery;

