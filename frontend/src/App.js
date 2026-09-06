import React, { useState, useEffect, useCallback } from 'react';
import './App.css';
import LiveFeed from './components/LiveFeed';
import TrendChart from './components/TrendChart';
import AlertPanel from './components/AlertPanel';
import MetricsPanel from './components/MetricsPanel';

const DEFAULT_METRICS = {
  defect_summary: {
    missing_hole: 12,
    mouse_bite: 8,
    open: 15,
    short: 24,
    spur: 7,
    spurious_copper: 11,
  },
  total_detections: 77,
  latest_rul: {
    equipment_id: 'EQP_001',
    health_score: 0.94,
    rul_cycles: 470,
    defect_count: 3,
  },
  health_trend: [0.99, 0.98, 0.97, 0.96, 0.95, 0.94],
};

function App() {
  const [equipmentId, setEquipmentId] = useState('EQP_001');
  const [metrics, setMetrics] = useState(DEFAULT_METRICS);
  const [connected, setConnected] = useState(false);

  const fetchMetrics = useCallback(async () => {
    try {
      const res = await fetch(`http://localhost:8000/metrics?equipment_id=${equipmentId}`);
      if (res.ok) {
        const data = await res.json();
        setMetrics(data);
        setConnected(true);
      } else {
        setConnected(false);
      }
    } catch (e) {
      setConnected(false);
    }
  }, [equipmentId]);

  useEffect(() => {
    fetchMetrics();
    const interval = setInterval(fetchMetrics, 5000);
    return () => clearInterval(interval);
  }, [fetchMetrics]);

  const handleDetectionComplete = () => {
    fetchMetrics();
  };

  return (
    <div className="dashboard-container">
      <header className="dashboard-header">
        <div className="header-brand">
          <span className="brand-icon">⚡</span>
          <div>
            <h1>PCB Smart Maintenance System</h1>
            <p className="subtitle">Real-Time Predictive Defect Detection & RUL Estimation</p>
          </div>
        </div>

        <div className="header-controls">
          <div className={`connection-status ${connected ? 'online' : 'offline'}`}>
            <span className="dot"></span>
            {connected ? 'Backend Online (Port 8000)' : 'Demo Mode (Offline)'}
          </div>

          <div className="equipment-selector">
            <label htmlFor="eqp-select">Machine:</label>
            <select
              id="eqp-select"
              value={equipmentId}
              onChange={(e) => setEquipmentId(e.target.value)}
            >
              <option value="EQP_001">EQP_001 (Line 1 SMT)</option>
              <option value="EQP_002">EQP_002 (Line 2 Reflow)</option>
              <option value="PI_EDGE_001">PI_EDGE_001 (Raspberry Pi 4)</option>
            </select>
          </div>
        </div>
      </header>

      <main className="dashboard-main">
        {/* KPI Metrics */}
        <section className="dashboard-section">
          <MetricsPanel metrics={metrics} />
        </section>

        {/* Alerts & Health Status */}
        <section className="dashboard-section">
          <AlertPanel metrics={metrics} />
        </section>

        {/* Live Feed & Defect Trends Grid */}
        <div className="dashboard-grid">
          <section className="panel-card feed-panel">
            <div className="panel-title">
              <h3>Live Optical Inspection</h3>
              <span className="badge">YOLOv8 + Domain Adapter</span>
            </div>
            <LiveFeed
              equipmentId={equipmentId}
              onDetectionComplete={handleDetectionComplete}
            />
          </section>

          <section className="panel-card chart-panel">
            <div className="panel-title">
              <h3>Defect Distribution (24h Frequency)</h3>
              <span className="badge">Class Alignment</span>
            </div>
            <TrendChart metrics={metrics} />
          </section>
        </div>
      </main>

      <footer className="dashboard-footer">
        <p>AI Smart Maintenance System · MAIT CSE AI/ML · Powered by YOLOv8, PyTorch MMD, FastAPI & ONNX</p>
      </footer>
    </div>
  );
}

export default App;

