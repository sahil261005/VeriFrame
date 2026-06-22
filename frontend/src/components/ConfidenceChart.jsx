import React from 'react';
import { ResponsiveContainer, AreaChart, Area, XAxis, YAxis, Tooltip, CartesianGrid } from 'recharts';

function ConfidenceChart({ report, duration }) {
  const frameExplanations = report?.frame_level_details || {};
  const flaggedTimestamps = Object.keys(frameExplanations).map(Number).sort((a, b) => a - b);

  // generate timeline data from 0s to duration
  const data = [];
  const step = 0.5; // step size in seconds
  
  for (let t = 0; t <= duration; t += step) {
    const timestamp = parseFloat(t.toFixed(1));
    // check if this timestamp was flagged (close match)
    const isFlagged = flaggedTimestamps.some(ft => Math.abs(ft - timestamp) < 0.25);
    
    data.push({
      time: `${timestamp}s`,
      score: isFlagged ? 0.85 : 0.12, // peak at flagged frames
    });
  }

  return (
    <div className="card" style={{ padding: '24px', width: '100%' }}>
      <h4 style={{ fontSize: '16px', fontWeight: '700', marginBottom: '16px' }}>Anomaly Detection Timeline</h4>
      <div style={{ width: '100%', height: '240px' }}>
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart data={data} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
            <defs>
              <linearGradient id="colorScore" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="var(--primary)" stopOpacity={0.4}/>
                <stop offset="95%" stopColor="var(--primary)" stopOpacity={0.0}/>
              </linearGradient>
            </defs>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
            <XAxis dataKey="time" stroke="var(--text-muted)" fontSize={11} tickLine={false} />
            <YAxis domain={[0, 1]} stroke="var(--text-muted)" fontSize={11} tickLine={false} />
            <Tooltip 
              contentStyle={{ 
                backgroundColor: 'var(--bg-secondary)', 
                borderColor: 'var(--border-color)',
                borderRadius: '8px',
                color: '#fff',
                fontSize: '13px'
              }}
            />
            <Area 
              type="monotone" 
              dataKey="score" 
              stroke="var(--primary)" 
              strokeWidth={2}
              fillOpacity={1} 
              fill="url(#colorScore)" 
            />
          </AreaChart>
        </ResponsiveContainer>
      </div>
      <div style={{ marginTop: '12px', fontSize: '12px', color: 'var(--text-muted)', textAlign: 'center' }}>
        Peaks indicate timestamps with concentrated visual/temporal anomalies flagged by the pipeline.
      </div>
    </div>
  );
}

export default ConfidenceChart;
