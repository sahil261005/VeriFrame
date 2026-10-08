import React from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { ScanFace, Plus, LogOut } from 'lucide-react';
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
        <Link to="/" className="logo">
          <span className="logo-mark"><ScanFace size={18} strokeWidth={2.2} /></span>
          <span style={{ display: 'flex', flexDirection: 'column', lineHeight: 1.15 }}>
            VeriFrame
            <span className="logo-sub">Multi-agent video forensics</span>
          </span>
        </Link>
        <div className="nav-links">
          {isAuthenticated ? (
            <>
              <Link to="/" className="btn btn-ghost" style={{ padding: '8px 12px', fontSize: '13px' }}>
                <Plus size={15} /> <span className="nav-label">New scan</span>
              </Link>
              {userEmail && <span className="nav-user">{userEmail}</span>}
              <button onClick={handleLogout} className="btn btn-secondary" style={{ padding: '8px 12px', fontSize: '13px' }}>
                <LogOut size={14} /> <span className="nav-label">Logout</span>
              </button>
            </>
          ) : (
            <>
              <Link to="/login" className="btn btn-secondary" style={{ padding: '8px 14px', fontSize: '13px' }}>Login</Link>
              <Link to="/register" className="btn" style={{ padding: '8px 14px', fontSize: '13px' }}>Register</Link>
            </>
          )}
        </div>
      </div>
    </nav>
  );
}

export default Navbar;
