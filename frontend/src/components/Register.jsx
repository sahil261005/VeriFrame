import React, { useState } from 'react';
import { useNavigate, Link } from 'react-router-dom';
import { TriangleAlert } from 'lucide-react';
import { authService } from '../api';
import AuthLayout from './AuthLayout';

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
    <AuthLayout>
      <div className="card" style={{ padding: '28px' }}>
        <h2 style={{ fontSize: '24px', fontWeight: 800, letterSpacing: '-0.025em' }}>Create your account</h2>
        <p className="muted" style={{ margin: '4px 0 22px', fontSize: '14px' }}>Free to use. Takes about ten seconds.</p>
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
            <input id="password" name="password" type="password" autoComplete="new-password" className="form-input" value={password} onChange={(e) => setPassword(e.target.value)} required minLength={8} />
              <div className="muted" style={{ fontSize: '12.5px', marginTop: '6px' }}>At least 8 characters.</div>
          </div>
            <div className="form-group">
              <label className="form-label" htmlFor="confirm">Confirm password</label>
              <input id="confirm" name="confirm" type="password" autoComplete="new-password" className="form-input" value={confirmPassword} onChange={(e) => setConfirmPassword(e.target.value)} required />
            </div>
          <button type="submit" className="btn" style={{ width: '100%', marginTop: '6px', padding: '12px' }} disabled={loading}>
            {loading ? 'Creating account…' : 'Create account'}
          </button>
        </form>
        <p style={{ textAlign: 'center', marginTop: '20px', fontSize: '14px', color: 'var(--ink-2)' }}>
          Already registered?{' '}
          <Link to="/login" style={{ fontWeight: 600 }}>Sign in</Link>
        </p>
      </div>
    </AuthLayout>
  );
}

export default Register;
