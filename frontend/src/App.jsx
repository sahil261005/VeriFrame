import React, { useState } from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import Navbar from './components/Navbar';
import Login from './components/Login';
import Register from './components/Register';
import Upload from './components/Upload';
import StatusFeed from './components/StatusFeed';
import ResultsDashboard from './components/ResultsDashboard';
import { authService } from './api';

// Route guard wrapper for authenticated endpoints
const ProtectedRoute = ({ children }) => {
  if (!authService.isAuthenticated()) {
    return <Navigate to="/login" replace />;
  }
  return children;
};

// Route guard wrapper for public auth pages
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
            {/* authentication paths */}
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

            {/* scan upload dashboard home path */}
            <Route path="/" element={
              <ProtectedRoute>
                {activeJobId ? (
                  <StatusFeed 
                    jobId={activeJobId} 
                    onAnalysisComplete={(id) => {
                      setActiveJobId(null);
                      // redirect cleanly using standard window location
                      window.location.href = `/analysis/${id}`;
                    }} 
                  />
                ) : (
                  <Upload onUploadSuccess={(id) => setActiveJobId(id)} />
                )}
              </ProtectedRoute>
            } />

            {/* results dashboard detailed analysis path */}
            <Route path="/analysis/:jobId" element={
              <ProtectedRoute>
                <ResultsDashboard />
              </ProtectedRoute>
            } />

            {/* default route fallback */}
            <Route path="*" element={<Navigate to="/" replace />} />
          </Routes>
        </main>
      </div>
    </Router>
  );
}

export default App;
