import { useState } from 'react';
import { Bell, AlertCircle, AlertTriangle, Info, CheckCircle2 } from 'lucide-react';
import { alerts } from '../data/mockDetections';
import './AlertPanel.css';

const FILTERS = ['all', 'critical', 'warning', 'info'];

const severityIcon = {
  critical: <AlertCircle size={12} />,
  warning: <AlertTriangle size={12} />,
  info: <Info size={12} />,
};

export default function AlertPanel() {
  const [activeFilter, setActiveFilter] = useState('all');

  const filtered = activeFilter === 'all'
    ? alerts
    : alerts.filter((a) => a.severity === activeFilter);

  return (
    <div className="glass-card alert-panel">
      <div className="glass-card__header">
        <div className="glass-card__title">
          <Bell size={16} />
          Alerts
        </div>
        <span className="alert-panel__count">{alerts.length} total</span>
      </div>

      <div className="alert-panel__tabs">
        {FILTERS.map((f) => (
          <button
            key={f}
            className={`alert-panel__tab ${activeFilter === f ? 'alert-panel__tab--active' : ''}`}
            onClick={() => setActiveFilter(f)}
          >
            {f}
          </button>
        ))}
      </div>

      <div className="alert-panel__list">
        {filtered.length === 0 ? (
          <div className="alert-panel__empty">
            <CheckCircle2 size={32} />
            No {activeFilter} alerts
          </div>
        ) : (
          filtered.map((alert, i) => (
            <div
              key={alert.id}
              className={`alert-item alert-item--${alert.severity}`}
              style={{ animationDelay: `${i * 0.05}s` }}
            >
              <div className="alert-item__header">
                <div className={`alert-item__severity alert-item__severity--${alert.severity}`}>
                  {severityIcon[alert.severity]}
                  {alert.severity}
                </div>
                <span className="alert-item__time">
                  {new Date(alert.timestamp).toLocaleTimeString('en-US', {
                    hour: '2-digit',
                    minute: '2-digit',
                    hour12: false,
                  })}
                </span>
              </div>
              <div className="alert-item__message">{alert.message}</div>
              <div className="alert-item__meta">
                <span className="alert-item__meta-tag">{alert.defectClass.replace(/_/g, ' ')}</span>
                <span className="alert-item__meta-tag">{alert.boardId}</span>
                <span className="alert-item__meta-tag">{(alert.confidence * 100).toFixed(0)}%</span>
              </div>
            </div>
          ))
        )}
      </div>
    </div>
  );
}
