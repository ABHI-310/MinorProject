import React from 'react';

function AlertPanel({ metrics }) {
  const healthScore = metrics?.latest_rul?.health_score ?? 0.96;
  const rulCycles = metrics?.latest_rul?.rul_cycles ?? 480;

  let alertLevel = 'green';
  let statusText = 'HEALTHY – Line Operating Within Tolerances';
  let actionPrompt = 'No immediate calibration or maintenance required.';

  if (healthScore < 0.40) {
    alertLevel = 'red';
    statusText = 'CRITICAL – Severe Drift / Immediate Maintenance Required';
    actionPrompt = 'Halt machine line EQP_001. Inspect solder nozzle, drill alignment, and optics.';
  } else if (healthScore < 0.70) {
    alertLevel = 'yellow';
    statusText = 'WARNING – Calibration Drift Detected';
    actionPrompt = 'Schedule inspection within next production shift. Monitor defect frequency.';
  }

  return (
    <div className={`alert-card alert-${alertLevel}`}>
      <div className="alert-header">
        <span className="status-indicator"></span>
        <h3 className="alert-title">{statusText}</h3>
      </div>
      <div className="alert-body">
        <div className="alert-stat">
          <span className="stat-label">Equipment Health</span>
          <span className="stat-value">{(healthScore * 100).toFixed(1)}%</span>
        </div>
        <div className="alert-stat">
          <span className="stat-label">Remaining Useful Life (RUL)</span>
          <span className="stat-value">{rulCycles} cycles</span>
        </div>
      </div>
      <div className="alert-action">
        <strong>Recommended Action:</strong> {actionPrompt}
      </div>
    </div>
  );
}

export default AlertPanel;

