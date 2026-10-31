import React, { useState, useEffect } from 'react';
import {
  RotateCw,
  Calendar,
  ChevronDown,
  User,
  Activity
} from 'lucide-react';
import { systemMetrics } from '../data/mockDetections';
import './Header.css';

export default function Header({ onSync }) {
  const [timeStr, setTimeStr] = useState('14:32:08 UTC');
  const [isSyncing, setIsSyncing] = useState(false);

  useEffect(() => {
    const updateTime = () => {
      const now = new Date();
      const hours = String(now.getUTCHours()).padStart(2, '0');
      const minutes = String(now.getUTCMinutes()).padStart(2, '0');
      const seconds = String(now.getUTCSeconds()).padStart(2, '0');
      setTimeStr(`${hours}:${minutes}:${seconds} UTC`);
    };
    updateTime();
    const interval = setInterval(updateTime, 1000);
    return () => clearInterval(interval);
  }, []);

  const handleSyncClick = () => {
    setIsSyncing(true);
    setTimeout(() => {
      setIsSyncing(false);
      if (onSync) onSync();
    }, 600);
  };

  return (
    <header className="top-header">
      {/* Left Title and Breadcrumbs */}
      <div className="top-header__left">
        <div className="top-header__icon-box">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" className="header-svg">
            <path d="M2 12h3l2-6 4 14 3-9 2 4 4-3h2" />
          </svg>
        </div>
        <div className="top-header__titles">
          <div className="top-header__title-row">
            <h1 className="top-header__title">Production Overview</h1>
            <span className="top-header__line-badge">{systemMetrics.activeLine}</span>
          </div>
          <p className="top-header__subtitle">
            Monitor PCB inspections and defect activity in real-time
          </p>
        </div>
      </div>

      {/* Right Controls & Status */}
      <div className="top-header__right">
        {/* System Online Badge */}
        <div className="top-header__status-badge">
          <span className="top-header__status-dot"></span>
          <span>SYSTEM ONLINE</span>
        </div>

        {/* Inference Speed */}
        <div className="top-header__metric-chip">
          <span className="top-header__metric-label">Inference:</span>
          <span className="top-header__metric-val">{systemMetrics.inferenceTime}</span>
        </div>

        {/* Clock */}
        <div className="top-header__time-chip">
          {timeStr}
        </div>

        {/* Sync Button */}
        <button
          className={`top-header__action-btn ${isSyncing ? 'top-header__action-btn--syncing' : ''}`}
          onClick={handleSyncClick}
          title="Sync latest inference data"
        >
          <RotateCw size={14} className={isSyncing ? 'spin-icon' : ''} />
          <span>Sync</span>
        </button>

        {/* Date Selector */}
        <div className="top-header__dropdown">
          <Calendar size={14} className="dropdown-icon" />
          <span className="dropdown-text">{systemMetrics.dateRange}</span>
          <ChevronDown size={14} className="dropdown-chevron" />
        </div>

        {/* Demo Mode Badge */}
        <span className="top-header__demo-pill">
          Demo Mode
        </span>

        {/* Profile Avatar */}
        <div className="top-header__avatar" title="Signed in as Quality Engineer">
          <User size={16} />
        </div>
      </div>
    </header>
  );
}
