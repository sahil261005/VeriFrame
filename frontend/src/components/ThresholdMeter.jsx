// score bar marked with the consensus engine's real cut-offs:
// below 0.30 can be AUTHENTIC, 0.30-0.48 is UNCERTAIN, 0.48 and above is MANIPULATED
const CUTS = [0.3, 0.48];

function ThresholdMeter({ score, color, compact = false }) {
  const pct = Math.max(0, Math.min(1, score)) * 100;
  return (
    <div>
      <div className="meter" style={compact ? { height: '6px' } : undefined}>
        <div className="meter-fill" style={{ width: `${Math.max(pct, 1.5)}%`, background: color }} />
        {CUTS.map((c) => <div key={c} className="meter-tick" style={{ left: `${c * 100}%` }} />)}
      </div>
      {!compact && (
        <div className="meter-scale">
          <span style={{ left: 0 }}>0</span>
          <span style={{ left: '30%' }}>.30</span>
          <span style={{ left: '48%' }}>.48</span>
          <span style={{ left: '100%' }}>1</span>
        </div>
      )}
    </div>
  );
}

export default ThresholdMeter;
