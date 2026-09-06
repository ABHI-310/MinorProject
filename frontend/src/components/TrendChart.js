import React from 'react';

function TrendChart({ metrics }) {
  if (!metrics) {
    return <div className="chart-placeholder">Loading defect distribution metrics...</div>;
  }

  const defectSummary = metrics.defect_summary || {};
  const entries = Object.entries(defectSummary);
  const maxCount = Math.max(1, ...entries.map(([_, v]) => v));

  return (
    <div className="trend-chart-card">
      <div className="chart-bars">
        {entries.map(([defectClass, count]) => {
          const heightPct = Math.min(100, Math.round((count / maxCount) * 100));
          return (
            <div key={defectClass} className="bar-column">
              <div className="bar-track">
                <div
                  className={`bar-fill bar-${defectClass}`}
                  style={{ height: `${Math.max(8, heightPct)}%` }}
                >
                  <span className="bar-tooltip">{count}</span>
                </div>
              </div>
              <div className="bar-label">{defectClass.replace('_', ' ')}</div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

export default TrendChart;

