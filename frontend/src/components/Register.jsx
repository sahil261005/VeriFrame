import React, { useState } from 'react';
import { useNavigate, Link } from 'react-router-dom';
import { authService } from '../api';

function Register() {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const navigate = useNavigate();

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError('');
    
    if (password !== confirmPassword) {
      setError('Passwords do not match');
      return;
    }

    setLoading(true);
    try {
      await authService.register(email, password);
      window.location.href = '/';
    } catch (err) {
      const detail = err.response?.data?.detail;
      if (detail) {
        if (typeof detail === 'string') {
          setError(detail);
        } else if (Array.isArray(detail)) {
          // Format validation errors list cleanly
          setError(detail.map(e => e.msg).join(', '));
        } else {
          setError(JSON.stringify(detail));
        }
      } else {
        setError('Registration failed. Try again.');
      }
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="auth-split-layout">
      <div className="auth-info-side">
        <h1 className="auth-tagline">
          Multi-Agent Deepfake & Video Manipulation Detection
        </h1>
        <p className="auth-description">
          An advanced, cooperative multi-agent platform designed to detect deepfakes, face mesh discrepancies, and temporal sequence inconsistencies.
        </p>
        
        <div className="auth-feature-list">
          <div className="auth-feature-item">
            <span className="auth-feature-icon">1</span>
            <div className="auth-feature-text">
              <div className="auth-feature-title">Visual Analysis Agent</div>
              Spatial frame checking utilizing deep learning classification models and high-frequency noise variance.
            </div>
          </div>
          <div className="auth-feature-item">
            <span className="auth-feature-icon">2</span>
            <div className="auth-feature-text">
              <div className="auth-feature-title">Temporal Analysis Agent</div>
              Tracks optical flow anomalies and matches facial landmarks dynamically to detect face-swaps.
            </div>
          </div>
          <div className="auth-feature-item">
            <span className="auth-feature-icon">3</span>
            <div className="auth-feature-text">
              <div className="auth-feature-title">Cognitive Reasoning Agent</div>
              Llama 4 Scout evaluates flagged frames, explaining precisely why a sample is suspect.
            </div>
          </div>
        </div>
      </div>

      <div className="auth-card-side">
        <div className="card">
          <h2 style={{ fontSize: '24px', fontWeight: '700', marginBottom: '20px', textAlign: 'center' }}>
            Create Account
          </h2>
          {error && (
            <div style={{ padding: '12px', borderRadius: '6px', backgroundColor: 'rgba(239,68,68,0.1)', border: '1px solid rgba(239,68,68,0.2)', color: '#ef4444', fontSize: '14px', marginBottom: '20px' }}>
              {error}
            </div>
          )}
          <form onSubmit={handleSubmit}>
            <div className="form-group">
              <label className="form-label">Email Address</label>
              <div>
                <input
                  type="email"
                  className="form-input"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  required
                />
              </div>
            </div>
            <div className="form-group">
              <label className="form-label">Password</label>
              <div>
                <input
                  type="password"
                  className="form-input"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  required
                />
              </div>
            </div>
            <div className="form-group">
              <label className="form-label">Confirm Password</label>
              <div>
                <input
                  type="password"
                  className="form-input"
                  value={confirmPassword}
                  onChange={(e) => setConfirmPassword(e.target.value)}
                  required
                />
              </div>
            </div>
            <button type="submit" className="btn" style={{ width: '100%', marginTop: '10px' }} disabled={loading}>
              {loading ? 'Creating account...' : 'Register'}
            </button>
          </form>
          <p style={{ textAlign: 'center', marginTop: '20px', fontSize: '14px', color: 'var(--text-secondary)' }}>
            Already have an account?{' '}
            <Link to="/login" style={{ color: 'var(--primary)', textDecoration: 'none', fontWeight: '500' }}>
              Login here
            </Link>
          </p>
        </div>
      </div>
    </div>
  );
}

export default Register;

