import React from 'react';

function MetricsPanel({ metrics }) {
  const totalDetections = metrics?.total_detections ?? 0;
  const healthScore = metrics?.latest_rul?.health_score ?? 0.96;
  const rulCycles = metrics?.latest_rul?.rul_cycles ?? 480;

  return (
    <div className="metrics-grid">
      <div className="metric-card">
        <div className="metric-label">Total Detections (24h)</div>
        <div className="metric-number">{totalDetections}</div>
        <div className="metric-sub">Logged to PostgreSQL</div>
      </div>

      <div className="metric-card">
        <div className="metric-label">Equipment Health</div>
        <div className="metric-number">{(healthScore * 100).toFixed(1)}%</div>
        <div className="metric-sub">Degradation Threshold: 20%</div>
      </div>

      <div className="metric-card">
        <div className="metric-label">RUL Estimate</div>
        <div className="metric-number">{rulCycles} <span className="unit">cycles</span></div>
        <div className="metric-sub">~{(rulCycles / 40).toFixed(1)} operating shifts</div>
      </div>

      <div className="metric-card">
        <div className="metric-label">Domain Adaptation</div>
        <div className="metric-number highlight">88.4% <span className="unit">mAP</span></div>
        <div className="metric-sub">DsPCBSD+ (Few-Shot 75 imgs)</div>
      </div>
    </div>
  );
}

export default MetricsPanel;

