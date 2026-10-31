import React from 'react';
import { Bell } from 'lucide-react';
import { usePipeline } from '../data/PipelineContext';
import './RecentAlerts.css';

export default function RecentAlerts({ onReviewAnomaly }) {
  const { recentAlerts } = usePipeline();
  return (
    <div className="ui-card alerts-card">
      {/* Header */}
      <div className="alerts-card__header">
        <div className="alerts-card__title-left">
          <Bell size={16} className="alerts-bell-icon" />
          <h2 className="alerts-card__title">Recent Alerts</h2>
        </div>
        <span className="alerts-pending-badge">{recentAlerts.length} recent</span>
      </div>

      {/* Alerts List */}
      <div className="alerts-card__list">
        {!recentAlerts.length && <p>No critical detections in recent history.</p>}
        {recentAlerts.map((alert) => (
          <div key={alert.id} className={`alert-box alert-box--${alert.severity}`}>
            <div className="alert-box__top">
              <span className={`alert-type-text alert-type-text--${alert.severity}`}>
                {alert.type}
              </span>
              <span className="alert-time-text">{alert.timeAgo}</span>
            </div>
            <p className="alert-desc">{alert.description}</p>
            <div className="alert-box__bottom">
              <span className="alert-ref-id">
                {alert.boardId || alert.location}
              </span>
              {alert.hasAction && (
                <button
                  className="alert-action-btn"
                  onClick={() => onReviewAnomaly && onReviewAnomaly(alert.inspectionRef)}
                >
                  {alert.actionText || 'Review Anomaly'}
                </button>
              )}
            </div>
          </div>
        ))}
      </div>

      {/* Footnote */}
      <p className="alerts-footnote">
        Short/open-circuit detections in the latest 100 inspections. These are model findings, not confirmed machine failures.
      </p>
    </div>
  );
}
