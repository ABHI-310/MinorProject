import React from 'react';
import { Bell } from 'lucide-react';
import { recentAlerts } from '../data/mockDetections';
import './RecentAlerts.css';

export default function RecentAlerts({ onReviewAnomaly }) {
  return (
    <div className="ui-card alerts-card">
      {/* Header */}
      <div className="alerts-card__header">
        <div className="alerts-card__title-left">
          <Bell size={16} className="alerts-bell-icon" />
          <h2 className="alerts-card__title">Recent Alerts</h2>
        </div>
        <span className="alerts-pending-badge">3 Pending</span>
      </div>

      {/* Alerts List */}
      <div className="alerts-card__list">
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
        Illustrative demonstration events. Bounding boxes and alerts do not indicate machine breakdown.
      </p>
    </div>
  );
}
