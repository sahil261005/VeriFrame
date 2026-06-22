import React, { useEffect, useState } from 'react';
import { Loader2, CheckCircle2, Circle } from 'lucide-react';
import { analysisService } from '../api';

function StatusFeed({ jobId, onAnalysisComplete }) {
  const [status, setStatus] = useState('processing');
  const [error, setError] = useState('');
  
  // mock steps representing the stages of the pipeline
  const [currentStep, setCurrentStep] = useState(0);

  const steps = [
    { label: 'Downscaling and preprocessing...', desc: 'Optimizing resolution to 480p and checking metadata.' },
    { label: 'Running parallel visual/temporal models...', desc: 'Analyzing frame sharpness and facial keypoints shifts.' },
    { label: 'Generating Gemini AI visual reasoning...', desc: 'Sending suspicious frames to Gemini 2.5 Flash.' },
    { label: 'Synthesizing consensus report...', desc: 'Recalculating weights and building final verdict.' }
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
              <Loader2 className="spinner" size={48} style={{ color: 'var(--primary)', animation: 'spin 1.5s linear infinite', marginBottom: '20px' }} />
              <h3 style={{ fontSize: '20px', fontWeight: '700', marginBottom: '8px' }}>Processing Video...</h3>
              <p style={{ fontSize: '14px', color: 'var(--text-secondary)' }}>
                Our LangGraph agent swarm is analyzing your file. Please do not close this window.
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
                  <div style={{ marginTop: '2px' }}>
                    {isCompleted ? (
                      <CheckCircle2 size={20} style={{ color: 'var(--success)' }} />
                    ) : isActive ? (
                      <Loader2 size={20} style={{ color: 'var(--primary)', animation: 'spin 1s linear infinite' }} />
                    ) : (
                      <Circle size={20} style={{ color: 'var(--text-muted)' }} />
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
