import React from 'react';
import {
  LayoutDashboard,
  Search,
  BarChart3,
  Bell,
  Settings,
  PanelLeftClose,
  PanelLeftOpen
} from 'lucide-react';
import './Sidebar.css';

export default function Sidebar({
  activeNav = 'overview',
  onSelectNav,
  isCollapsed = false,
  onToggleCollapse
}) {
  return (
    <aside className={`sidebar ${isCollapsed ? 'sidebar--collapsed' : ''}`}>
      {/* Brand Header */}
      <div className="sidebar__brand">
        <div className="sidebar__brand-main">
          <div className="sidebar__logo-icon" title="PCB Sentinel">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" className="logo-svg">
              <path d="M2 12h3l2-6 4 14 3-9 2 4 4-3h2" />
            </svg>
          </div>
          {!isCollapsed && (
            <div className="sidebar__brand-text">
              <div className="sidebar__brand-title">
                <span>PCB Sentinel</span>
                <span className="sidebar__version-badge">v2.4 Pro</span>
              </div>
              <div className="sidebar__brand-subtitle">AI Smart Maintenance</div>
            </div>
          )}
        </div>
        <button
          className="sidebar__toggle-btn"
          onClick={onToggleCollapse}
          title={isCollapsed ? 'Extend Sidebar' : 'Hide Sidebar'}
          aria-label={isCollapsed ? 'Extend Sidebar' : 'Hide Sidebar'}
        >
          {isCollapsed ? <PanelLeftOpen size={16} /> : <PanelLeftClose size={16} />}
        </button>
      </div>

      {/* Nav Section Header */}
      {!isCollapsed && <div className="sidebar__section-title">COMMAND CENTER</div>}

      <nav className="sidebar__nav">
        <button
          className={`sidebar__nav-item ${activeNav === 'overview' ? 'sidebar__nav-item--active' : ''}`}
          onClick={() => onSelectNav && onSelectNav('overview')}
          title="Overview"
        >
          <div className="sidebar__nav-left">
            <LayoutDashboard size={18} />
            {!isCollapsed && <span>Overview</span>}
          </div>
        </button>

        <button
          className={`sidebar__nav-item ${activeNav === 'inspections' ? 'sidebar__nav-item--active' : ''}`}
          onClick={() => onSelectNav && onSelectNav('inspections')}
          title="Inspections (1,284)"
        >
          <div className="sidebar__nav-left">
            <Search size={18} />
            {!isCollapsed && <span>Inspections</span>}
          </div>
          {!isCollapsed ? (
            <span className="sidebar__count-badge">1,284</span>
          ) : (
            <span className="sidebar__collapsed-dot"></span>
          )}
        </button>

        <button
          className={`sidebar__nav-item ${activeNav === 'analytics' ? 'sidebar__nav-item--active' : ''}`}
          onClick={() => onSelectNav && onSelectNav('analytics')}
          title="Defect Analytics"
        >
          <div className="sidebar__nav-left">
            <BarChart3 size={18} />
            {!isCollapsed && <span>Defect Analytics</span>}
          </div>
        </button>

        <button
          className={`sidebar__nav-item ${activeNav === 'alerts' ? 'sidebar__nav-item--active' : ''}`}
          onClick={() => onSelectNav && onSelectNav('alerts')}
          title="Alerts (8 active)"
        >
          <div className="sidebar__nav-left">
            <Bell size={18} />
            {!isCollapsed && <span>Alerts</span>}
          </div>
          {!isCollapsed ? (
            <span className="sidebar__alert-badge">8</span>
          ) : (
            <span className="sidebar__collapsed-alert-dot"></span>
          )}
        </button>

        <button
          className={`sidebar__nav-item ${activeNav === 'settings' ? 'sidebar__nav-item--active' : ''}`}
          onClick={() => onSelectNav && onSelectNav('settings')}
          title="Settings"
        >
          <div className="sidebar__nav-left">
            <Settings size={18} />
            {!isCollapsed && <span>Settings</span>}
          </div>
        </button>
      </nav>

      {/* System Status Footers */}
      <div className="sidebar__footer">
        <div className="sidebar__status-row" title="FastAPI Core: Connected">
          <div className="sidebar__status-left">
            <span className="status-dot status-dot--online"></span>
            {!isCollapsed && <span>FastAPI Core</span>}
          </div>
          {!isCollapsed && (
            <span className="sidebar__status-val sidebar__status-val--green">Connected</span>
          )}
        </div>

        <div className="sidebar__status-row" title="PostgreSQL DB: Optimal">
          <div className="sidebar__status-left">
            <span className="status-dot status-dot--online"></span>
            {!isCollapsed && <span>PostgreSQL DB</span>}
          </div>
          {!isCollapsed && (
            <span className="sidebar__status-val sidebar__status-val--green">Optimal</span>
          )}
        </div>
      </div>
    </aside>
  );
}
