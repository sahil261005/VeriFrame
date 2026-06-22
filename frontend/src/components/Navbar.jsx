import React from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { authService } from '../api';

function Navbar() {
  const navigate = useNavigate();
  const isAuthenticated = authService.isAuthenticated();
  const userEmail = authService.getUserEmail();

  const handleLogout = () => {
    authService.logout();
    navigate('/login');
  };

  return (
    <nav className="navbar">
      <div className="nav-content">
        <Link to="/" className="logo" style={{ textDecoration: 'none', color: 'var(--text-primary)', fontWeight: '700', fontSize: '20px' }}>
          VeriFrame
        </Link>
        <div className="nav-links">
          {isAuthenticated ? (
            <>
              <Link to="/" style={{ fontSize: '15px', fontWeight: '500', marginRight: '15px', textDecoration: 'none', color: 'var(--primary)' }}>
                New Scan
              </Link>
              <span className="nav-user" style={{ marginRight: '10px' }}>{userEmail}</span>
              <button onClick={handleLogout} className="btn btn-secondary" style={{ padding: '8px 16px', fontSize: '13px' }}>
                Logout
              </button>
            </>
          ) : (
            <>
              <Link to="/login" className="btn btn-secondary" style={{ padding: '8px 16px', fontSize: '13px' }}>Login</Link>
              <Link to="/register" className="btn" style={{ padding: '8px 16px', fontSize: '13px' }}>Register</Link>
            </>
          )}
        </div>
      </div>
    </nav>
  );
}

export default Navbar;

