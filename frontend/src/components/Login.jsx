import React, { useState } from 'react';
import { useNavigate, Link } from 'react-router-dom';
import { TriangleAlert } from 'lucide-react';
import { authService } from '../api';
import AuthLayout from './AuthLayout';

function Login() {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const navigate = useNavigate();

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError('');
    setLoading(true);
    try {
      await authService.login(email, password);
      window.location.href = '/';
    } catch (err) {
      const detail = err.response?.data?.detail;
      if (detail) {
        if (typeof detail === 'string') {
          setError(detail);
        } else if (Array.isArray(detail)) {
          setError(detail.map(e => e.msg).join(', '));
        } else {
          setError(JSON.stringify(detail));
        }
      } else {
        setError('Invalid email or password');
      }
    } finally {
      setLoading(false);
    }
  };

  return (
    <AuthLayout>
      <div className="card" style={{ padding: '28px' }}>
        <h2 style={{ fontSize: '24px', fontWeight: 800, letterSpacing: '-0.025em' }}>Sign in</h2>
        <p className="muted" style={{ margin: '4px 0 22px', fontSize: '14px' }}>Welcome back. Pick up where you left off.</p>
        {error && (
          <div className="banner banner-danger" style={{ marginBottom: '16px' }} role="alert">
            <TriangleAlert size={18} />
            <span>{error}</span>
          </div>
        )}
        <form onSubmit={handleSubmit}>
          <div className="form-group">
            <label className="form-label" htmlFor="email">Email address</label>
            <input id="email" name="email" type="email" autoComplete="email" className="form-input" value={email} onChange={(e) => setEmail(e.target.value)} required />
          </div>
          <div className="form-group">
            <label className="form-label" htmlFor="password">Password</label>
            <input id="password" name="password" type="password" autoComplete="current-password" className="form-input" value={password} onChange={(e) => setPassword(e.target.value)} required />
          </div>
          <button type="submit" className="btn" style={{ width: '100%', marginTop: '6px', padding: '12px' }} disabled={loading}>
            {loading ? 'Signing in…' : 'Sign in'}
          </button>
        </form>
        <p style={{ textAlign: 'center', marginTop: '20px', fontSize: '14px', color: 'var(--ink-2)' }}>
          New to VeriFrame?{' '}
          <Link to="/register" style={{ fontWeight: 600 }}>Create an account</Link>
        </p>
      </div>
    </AuthLayout>
  );
}

export default Login;
