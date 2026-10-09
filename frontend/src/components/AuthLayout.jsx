import React from 'react';
import { ScanFace, Activity, AudioWaveform, BadgeCheck, BrainCircuit } from 'lucide-react';
import ThresholdMeter from './ThresholdMeter';

// just example numbers for the landing side, not real results
const SAMPLE = [
  { icon: ScanFace, name: 'Face-swap', score: 0.12 },
  { icon: Activity, name: 'Motion', score: 0.31 },
  { icon: AudioWaveform, name: 'Audio', score: 0.62 },
  { icon: BrainCircuit, name: 'Vision LLM', score: 0.93 },
];
const scoreColor = (s) => (s >= 0.48 ? 'var(--bad)' : s >= 0.3 ? 'var(--warn)' : 'var(--ok)');

function AuthLayout({ children }) {
  return (
    <div className="auth-split-layout">
      <div className="auth-info-side">
        <span className="label">Multi-agent video forensics</span>
        <h1 className="auth-tagline">Find out if a video was <em>faked</em>.</h1>
        <p className="auth-description">
          Upload a clip of up to 30 seconds. Independent checks look at faces, motion, audio and file
          credentials, then a vision AI explains what looks wrong, frame by frame.
        </p>

        <div className="specimen" aria-label="Example report">
          <div className="specimen-head">
            <span className="label">Example report</span>
            <span className="badge badge-manipulated">Manipulated</span>
          </div>
          <div className="specimen-body">
            <div className="specimen-score">
              <span className="verdict-score" style={{ '--verdict-color': 'var(--bad)' }}>
                91<small>/ 100</small>
              </span>
              <span className="label">Manipulation score</span>
            </div>
            <div className="specimen-rows">
              {SAMPLE.map(({ icon: Icon, name, score }) => (
                <div key={name} className="specimen-row">
                  <span className="specimen-name"><Icon size={14} /> {name}</span>
                  <ThresholdMeter score={score} color={scoreColor(score)} compact />
                  <span className="mono specimen-num">{score.toFixed(2)}</span>
                </div>
              ))}
            </div>
          </div>
          <div className="specimen-note">
            <BadgeCheck size={14} />
            <span>Signed &ldquo;made by AI&rdquo; labels inside the file (C2PA) are checked too, and count as strong evidence.</span>
          </div>
        </div>

        <p className="auth-fineprint">
          The video file is deleted as soon as the analysis finishes. Only the report and a few frames are kept.
        </p>
      </div>

      <div className="auth-card-side">{children}</div>
    </div>
  );
}

export default AuthLayout;
