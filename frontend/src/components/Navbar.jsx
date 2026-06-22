import React from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { Shield, LogOut, Video } from 'lucide-react';
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
          <Shield size={22} />
          Veri<span>Frame</span>
        </Link>
        <div className="nav-links">
          {isAuthenticated ? (
            <>
              <Link to="/" className="logo" style={{ fontSize: '15px', fontWeight: '500', marginRight: '15px' }}>
                <Video size={18} />
                New Scan
              </Link>
              <span className="nav-user" style={{ marginRight: '10px' }}>{userEmail}</span>
              <button onClick={handleLogout} className="btn btn-secondary" style={{ padding: '8px 16px', fontSize: '13px' }}>
                <LogOut size={16} />
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
