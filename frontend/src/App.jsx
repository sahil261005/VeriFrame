import React, { useState } from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import Navbar from './components/Navbar';
import Login from './components/Login';
import Register from './components/Register';
import Upload from './components/Upload';
import StatusFeed from './components/StatusFeed';
import ResultsDashboard from './components/ResultsDashboard';
import { authService } from './api';

// only lets logged in users through
const ProtectedRoute = ({ children }) => {
  if (!authService.isAuthenticated()) {
    return <Navigate to="/login" replace />;
  }
  return children;
};

// if youre already logged in theres no point showing login/register
const PublicRoute = ({ children }) => {
  if (authService.isAuthenticated()) {
    return <Navigate to="/" replace />;
  }
  return children;
};

function App() {
  const [activeJobId, setActiveJobId] = useState(null);
  
  return (
    <Router>
      <div className="app-container">
        <Navbar />
        <main className="main-content">
          <Routes>
            {/* login and register */}
            <Route path="/login" element={
              <PublicRoute>
                <Login />
              </PublicRoute>
            } />
            <Route path="/register" element={
              <PublicRoute>
                <Register />
              </PublicRoute>
            } />

            {/* home page, upload a video or watch the current scan */}
            <Route path="/" element={
              <ProtectedRoute>
                {activeJobId ? (
                  <StatusFeed 
                    jobId={activeJobId} 
                    onAnalysisComplete={(id) => {
                      setActiveJobId(null);
                    }} 
                  />
                ) : (
                  <Upload onUploadSuccess={(id) => setActiveJobId(id)} />
                )}
              </ProtectedRoute>
            } />

            {/* results for one scan */}
            <Route path="/analysis/:jobId" element={
              <ProtectedRoute>
                <ResultsDashboard />
              </ProtectedRoute>
            } />

            {/* anything else just goes home */}
            <Route path="*" element={<Navigate to="/" replace />} />
          </Routes>
        </main>
      </div>
    </Router>
  );
}

export default App;
