import React, { useEffect, useState } from 'react';
import { analysisService } from '../api';

function StatusFeed({ jobId, onAnalysisComplete }) {
  const [status, setStatus] = useState('processing');
  const [error, setError] = useState('');
  
  // mock steps representing the stages of the pipeline
  const [currentStep, setCurrentStep] = useState(0);

  const steps = [
    { label: 'Decompressing and segmenting media...', desc: 'Extracting video keyframes and validating stream container.' },
    { label: 'Running visual artifact inspection...', desc: 'Analyzing color channel noise residuals and spatial anomalies.' },
    { label: 'Analyzing motion and temporal flow...', desc: 'Tracking facial landmarks and movement continuity between adjacent frames.' },
    { label: 'Generating consensus score...', desc: 'Calculating weighted averages from visual and temporal detectors.' }
  ];

  useEffect(() => {
    let intervalId;
    let stepTimer;

    // cycle through mock steps to give visual feedback to the user
    stepTimer = setInterval(() => {
      setCurrentStep((prev) => {
        if (prev < steps.length - 1) {
          return prev + 1;
        }
        return prev;
      });
    }, 4000);

    const checkStatus = async () => {
      try {
        const data = await analysisService.getAnalysis(jobId);
        setStatus(data.status);
        
        if (data.status === 'completed') {
          clearInterval(intervalId);
          clearInterval(stepTimer);
          setCurrentStep(steps.length); // mark all completed
          // brief delay so the user sees everything completed before routing
          setTimeout(() => {
            onAnalysisComplete(jobId);
          }, 1000);
        } else if (data.status === 'failed') {
          clearInterval(intervalId);
          clearInterval(stepTimer);
          setError('Video analysis pipeline encountered an error.');
        }
      } catch (err) {
        console.error('Error polling status:', err);
      }
    };

    // run initial check and start interval
    checkStatus();
    intervalId = setInterval(checkStatus, 2500);

    return () => {
      clearInterval(intervalId);
      clearInterval(stepTimer);
    };
  }, [jobId]);

  return (
    <div style={{ maxWidth: '600px', width: '100%', margin: '60px auto 0 auto' }}>
      <div className="card" style={{ padding: '40px', overflow: 'hidden' }}>
        <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', textAlign: 'center', marginBottom: '40px' }}>
          {error ? (
            <div style={{ color: 'var(--danger)', fontSize: '18px', fontWeight: '600' }}>Analysis Failed</div>
          ) : (
            <>
              <div style={{ fontSize: '14px', fontWeight: '600', color: 'var(--text-secondary)', marginBottom: '15px' }}>
                Scanning file...
              </div>
              <h3 style={{ fontSize: '20px', fontWeight: '700', marginBottom: '8px' }}>Processing Video...</h3>
              <p style={{ fontSize: '14px', color: 'var(--text-secondary)' }}>
                Running localized computer vision classification models. Please do not close this page.
              </p>
            </>
          )}
        </div>

        {error ? (
          <div style={{ padding: '16px', borderRadius: '8px', backgroundColor: 'rgba(239, 68, 68, 0.1)', border: '1px solid rgba(239, 68, 68, 0.2)', color: 'var(--danger)', fontSize: '14px', textAlign: 'center' }}>
            {error}
          </div>
        ) : (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
            {steps.map((step, idx) => {
              const isCompleted = idx < currentStep;
              const isActive = idx === currentStep;

              return (
                <div key={idx} style={{ display: 'flex', gap: '16px', alignItems: 'flex-start', opacity: isCompleted || isActive ? 1 : 0.4 }}>
                  <div style={{ marginTop: '2px', fontFamily: 'monospace', fontSize: '12px', fontWeight: '700', width: '80px', flexShrink: 0 }}>
                    {isCompleted ? (
                      <span style={{ color: 'var(--success)' }}>[ DONE ]</span>
                    ) : isActive ? (
                      <span style={{ color: 'var(--primary)' }}>[ RUNNING ]</span>
                    ) : (
                      <span style={{ color: 'var(--text-muted)' }}>[ PENDING ]</span>
                    )}
                  </div>
                  <div>
                    <div style={{ fontSize: '15px', fontWeight: '600', color: 'var(--text-primary)' }}>
                      {step.label}
                    </div>
                    <div style={{ fontSize: '13px', color: 'var(--text-secondary)', marginTop: '4px' }}>
                      {step.desc}
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
}

export default StatusFeed;


