import React, { useState, useMemo } from 'react';
import {
  Bell,
  AlertTriangle,
  AlertCircle,
  CheckCircle2,
  Clock,
  Download,
  Filter,
  Search,
  SlidersHorizontal,
  ChevronDown,
  Shield,
  Phone,
  Radio,
  ExternalLink,
  Plus,
  ArrowUpRight,
  UserCheck,
  Check,
  Wrench,
  Zap
} from 'lucide-react';
import './AlertsPage.css';

export default function AlertsPage({ onNavigateToInspection, onBackToOverview }) {
  const [searchQuery, setSearchQuery] = useState('');
  const [filterQuickset, setFilterQuickset] = useState('all'); // 'all' | 'critical' | 'warnings' | 'line01' | 'line04' | 'acknowledged'
  const [severityFilter, setSeverityFilter] = useState('all');
  const [toastMessage, setToastMessage] = useState(null);

  // Interlock toggle states
  const [interlocks, setInterlocks] = useState({
    autoPause: true,
    notifyDispatch: true,
    slackPager: true,
  });

  // Alert items state for interactive acknowledgement
  const [alertsList, setAlertsList] = useState([
    {
      id: 'alt-001',
      severity: 'critical',
      severityLabel: 'CRITICAL P1',
      ruleCode: 'Rule #ALT-SH-04',
      line: 'SMT Line 01 • Station Alpha',
      lineKey: 'line01',
      timeAgo: '4 mins ago',
      timestamp: '14:28:12 UTC',
      title: 'Repeated Micro-Shorts Detected on IC-U7 Footprint',
      boardId: '#PCB-2025-0842',
      confidence: '98.2%',
      thumbnail: '/roi_solder_bridge.jpg',
      tagText: 'SHORT',
      camLabel: 'AOI CAM 02',
      triggerText: '>3 short circuits in consecutive 10 boards',
      recommendedAction: 'Immediate Stencil Aperture Inspection & Solder Paste Dispenser Calibration required. Automatic line slow-down active.',
      metaPills: ['BGA Pitch: 0.4mm', 'Tech: Unassigned'],
      actionType: 'dispatch',
      acknowledged: false,
    },
    {
      id: 'alt-002',
      severity: 'critical',
      severityLabel: 'CRITICAL P1',
      ruleCode: 'Rule #ALT-OP-01',
      line: 'SMT Line 01 • Station Beta',
      lineKey: 'line01',
      timeAgo: '19 mins ago',
      timestamp: '14:13:20 UTC',
      title: 'Consecutive Open Circuit Anomaly on Layer 2 Bus',
      boardId: '#PCB-2025-0838',
      confidence: '94.8%',
      thumbnail: '/prediction_samples/pred_01_open_circuit_12_jpg.rf.640be935ceb2f18fbee221d8cf56bad3.jpg',
      tagText: 'OPEN TRACE',
      camLabel: 'AOI CAM 01',
      triggerText: 'Broken trace detected across primary 3.3V power plane',
      recommendedAction: 'Check pre-reflow pick-and-place nozzle pressure and examine etching tank acid balance for over-etching drift.',
      metaPills: ['Trace: L2-PWR-NET'],
      actionType: 'escalate',
      acknowledged: false,
    },
    {
      id: 'alt-003',
      severity: 'warning',
      severityLabel: 'WARNING P2',
      ruleCode: 'Rule #DRF-CU-09',
      line: 'SMT Line 04 • Etch Inspection',
      lineKey: 'line04',
      timeAgo: '42 mins ago',
      timestamp: '13:50:08 UTC',
      title: 'Spurious Copper Flakes Exceeding IPC Class 3 Tolerance',
      boardId: '#PCB-2025-0831',
      confidence: '91.7%',
      thumbnail: '/roi_copper_spur.jpg',
      tagText: 'SPURIOUS COPPER',
      triggerRule: 'Drift Warning • Etching bath degradation • Particle Size: 0.12mm (Max 0.08mm)',
      detailText: 'Non-bridging isolated copper sliver in isolation channel. Yield impact low, but trend indicates rinse bath contamination.',
      footerTag: 'IPC Class 3 Deviation: Minor',
      acknowledged: false,
    },
    {
      id: 'alt-004',
      severity: 'warning',
      severityLabel: 'WARNING P2',
      ruleCode: 'Rule #MB-RT-03',
      line: 'SMT Line 04 • Routing Unit',
      lineKey: 'line04',
      timeAgo: '1 hour ago',
      timestamp: '13:32:00 UTC',
      title: 'Mouse Bite Edge Roughness Deviation on Panel Border',
      boardId: '#PCB-2025-0824',
      confidence: '88.5%',
      thumbnail: '/prediction_samples/pred_01_mouse_bite_05_jpg.rf.f4cded1996897893b5e0216f9a365cb8.jpg',
      tagText: 'MOUSE BITE',
      detailText: 'Breakaway tab perforation burr exceeds 0.25mm threshold. Routing bit wear at 78%.',
      footerTag: 'Tool Life: 412 / 500 panels',
      acknowledged: false,
    },
    {
      id: 'alt-005',
      severity: 'acknowledged',
      severityLabel: 'ACKNOWLEDGED',
      ruleCode: 'Cleared',
      line: 'SMT Line 01',
      lineKey: 'line01',
      timeAgo: '2 hours ago',
      timestamp: '13:10:00 UTC',
      title: 'Missing Solder Mask Hole / Pad Bridging Risk',
      boardId: '#PCB-2025-0810',
      clearedNote: 'Acknowledged by Operator J. Carter at 13:18 UTC',
      quoteText: '"Cleared after manual stencil wipe and isopropyl rinse. Test coupon passed."',
      footerTag: 'Yield Normal',
      acknowledged: true,
    }
  ]);

  const showToast = (msg) => {
    setToastMessage(msg);
    setTimeout(() => setToastMessage(null), 3000);
  };

  const handleAcknowledge = (id, title) => {
    setAlertsList((prev) =>
      prev.map((a) => (a.id === id ? { ...a, acknowledged: true, severity: 'acknowledged' } : a))
    );
    showToast(`Alert Acknowledged: ${title}`);
  };

  const handleAcknowledgeAllNonCritical = () => {
    setAlertsList((prev) =>
      prev.map((a) => (a.severity !== 'critical' ? { ...a, acknowledged: true, severity: 'acknowledged' } : a))
    );
    showToast('All 14 Warning & Info alerts acknowledged');
  };

  const handleDispatchTech = (boardId) => {
    showToast(`Rework Tech dispatched to SMT Line 01 for board ${boardId}`);
  };

  const handleEscalateLead = (boardId) => {
    showToast(`Incident on ${boardId} escalated to Shift Lead Sarah Chen, PE`);
  };

  const filteredAlerts = useMemo(() => {
    return alertsList.filter((alert) => {
      // Search
      const matchesSearch =
        alert.title.toLowerCase().includes(searchQuery.toLowerCase()) ||
        alert.boardId.toLowerCase().includes(searchQuery.toLowerCase()) ||
        alert.line.toLowerCase().includes(searchQuery.toLowerCase());

      // Quickset filter
      let matchesQuickset = true;
      if (filterQuickset === 'critical') matchesQuickset = alert.severity === 'critical';
      else if (filterQuickset === 'warnings') matchesQuickset = alert.severity === 'warning';
      else if (filterQuickset === 'line01') matchesQuickset = alert.lineKey === 'line01';
      else if (filterQuickset === 'line04') matchesQuickset = alert.lineKey === 'line04';
      else if (filterQuickset === 'acknowledged') matchesQuickset = alert.acknowledged === true;

      return matchesSearch && matchesQuickset;
    });
  }, [alertsList, searchQuery, filterQuickset]);

  return (
    <div className="alerts-page">
      {/* Toast Notification */}
      {toastMessage && (
        <div className="alerts-toast">
          <CheckCircle2 size={16} />
          <span>{toastMessage}</span>
        </div>
      )}

      {/* Top Header & Breadcrumbs */}
      <div className="alerts-page-topbar">
        <div className="alerts-page-topbar__left">
          <nav className="alerts-breadcrumbs">
            <button className="breadcrumb-link" onClick={onBackToOverview}>COMMAND CENTER</button>
            <span className="breadcrumb-sep">/</span>
            <span className="breadcrumb-current">ALERTS & INCIDENTS</span>
            <span className="critical-header-badge">8 CRITICAL UNRESOLVED</span>
          </nav>

          <h1 className="alerts-page-title">System Alerts & Incident Center</h1>
          <p className="alerts-page-subtitle">
            Real-time notification triage, threshold breaches, and defect rule triggers across automated SMT production lines.
          </p>
        </div>

        <div className="alerts-page-topbar__right">
          <button className="topbar-btn topbar-btn--blue" onClick={handleAcknowledgeAllNonCritical}>
            <Check size={14} />
            <span>Acknowledge All Non-Critical</span>
          </button>

          <button className="topbar-btn topbar-btn--blue" onClick={() => showToast('Automation rule matrix opened')}>
            <SlidersHorizontal size={14} />
            <span>Configure Rules</span>
          </button>

          <button className="topbar-btn topbar-btn--brown" onClick={() => window.print()}>
            <Download size={14} />
            <span>Export Incident Log</span>
          </button>
        </div>
      </div>

      {/* 4 KPI Summary Cards */}
      <div className="alerts-kpi-grid">
        {/* Card 1 */}
        <div className="ui-card alerts-kpi-card">
          <div className="alerts-kpi-card__top">
            <span className="kpi-label">CRITICAL UNRESOLVED</span>
            <div className="kpi-icon-box kpi-icon-box--red">
              <AlertTriangle size={15} />
            </div>
          </div>
          <div className="kpi-val-row">
            <span className="kpi-num kpi-num--red">08</span>
            <span className="kpi-badge kpi-badge--red">+2 in last hr</span>
          </div>
          <div className="kpi-footer-row">
            <span className="kpi-sub-text">Requires machine interlock check</span>
            <span className="kpi-accent-text kpi-accent-text--red">Tier 1 Escalation</span>
          </div>
        </div>

        {/* Card 2 */}
        <div className="ui-card alerts-kpi-card">
          <div className="alerts-kpi-card__top">
            <span className="kpi-label">WARNINGS (YIELD DRIFT)</span>
            <div className="kpi-icon-box kpi-icon-box--amber">
              <AlertCircle size={15} />
            </div>
          </div>
          <div className="kpi-val-row">
            <span className="kpi-num">14</span>
            <span className="kpi-badge kpi-badge--blue">P2 Pending</span>
          </div>
          <div className="kpi-footer-row">
            <span className="kpi-sub-text">Requires operator signoff</span>
            <span className="kpi-accent-text kpi-accent-text--gray">Avg 12m delay</span>
          </div>
        </div>

        {/* Card 3 */}
        <div className="ui-card alerts-kpi-card">
          <div className="alerts-kpi-card__top">
            <span className="kpi-label">MEAN TIME TO ACK (MTTA)</span>
            <div className="kpi-icon-box kpi-icon-box--green">
              <Clock size={15} />
            </div>
          </div>
          <div className="kpi-val-row">
            <span className="kpi-num">4.2</span>
            <span className="kpi-unit">min</span>
            <span className="kpi-badge kpi-badge--green">18% faster</span>
          </div>
          <div className="kpi-footer-row">
            <span className="kpi-sub-text">Target SLA: &lt; 5.0 mins</span>
            <span className="kpi-accent-text kpi-accent-text--green">Compliant</span>
          </div>
        </div>

        {/* Card 4 */}
        <div className="ui-card alerts-kpi-card">
          <div className="alerts-kpi-card__top">
            <span className="kpi-label">ACTIVE LINES FLAGGED</span>
            <div className="kpi-icon-box kpi-icon-box--purple">
              <Radio size={15} />
            </div>
          </div>
          <div className="kpi-val-row">
            <span className="kpi-num">2</span>
            <span className="kpi-unit">/ 6</span>
            <span className="kpi-badge kpi-badge--gray">33% active</span>
          </div>
          <div className="kpi-footer-row">
            <div className="lines-dots">
              <span className="line-dot-red"></span>
              <span className="line-text">SMT-01</span>
              <span className="line-dot-red"></span>
              <span className="line-text">SMT-04</span>
            </div>
            <span className="kpi-accent-text kpi-accent-text--gray">In Spectation</span>
          </div>
        </div>
      </div>

      {/* Search & Filter Bar */}
      <div className="ui-card alerts-filter-card">
        <div className="alerts-filter-main">
          <div className="alerts-search-input-wrap">
            <Search size={14} className="alerts-search-icon" />
            <input
              type="text"
              placeholder="Search alerts by ID, board serial, line, or defect class..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="alerts-search-field"
            />
          </div>

          <div className="alerts-dropdown-selects">
            <div className="alerts-select-pill">
              <span>All Severities</span>
              <ChevronDown size={13} />
            </div>
            <div className="alerts-select-pill">
              <span>All Defect Classes</span>
              <ChevronDown size={13} />
            </div>
            <div className="alerts-select-pill">
              <span>Last 24 Hours</span>
              <ChevronDown size={13} />
            </div>
          </div>
        </div>

        {/* Quickset Filter Tabs */}
        <div className="alerts-quickset-row">
          <span className="quickset-label">FILTER QUICKSET:</span>
          <div className="quickset-pills">
            <button
              className={`quickset-pill ${filterQuickset === 'all' ? 'quickset-pill--active' : ''}`}
              onClick={() => setFilterQuickset('all')}
            >
              All (22)
            </button>
            <button
              className={`quickset-pill quickset-pill--red ${filterQuickset === 'critical' ? 'quickset-pill--active-red' : ''}`}
              onClick={() => setFilterQuickset('critical')}
            >
              Critical (8)
            </button>
            <button
              className={`quickset-pill quickset-pill--amber ${filterQuickset === 'warnings' ? 'quickset-pill--active-amber' : ''}`}
              onClick={() => setFilterQuickset('warnings')}
            >
              Warnings (14)
            </button>
            <button
              className={`quickset-pill ${filterQuickset === 'line01' ? 'quickset-pill--active-line' : ''}`}
              onClick={() => setFilterQuickset('line01')}
            >
              SMT Line 01 (12)
            </button>
            <button
              className={`quickset-pill ${filterQuickset === 'line04' ? 'quickset-pill--active-line' : ''}`}
              onClick={() => setFilterQuickset('line04')}
            >
              SMT Line 04 (10)
            </button>
            <button
              className={`quickset-pill ${filterQuickset === 'acknowledged' ? 'quickset-pill--active-line' : ''}`}
              onClick={() => setFilterQuickset('acknowledged')}
            >
              Acknowledged (1)
            </button>
          </div>
        </div>
      </div>

      {/* 2-Column Main Layout: Feed (Left) & Controls/Shift (Right) */}
      <div className="alerts-main-grid">
        {/* Left Column: Alerts Feed Cards */}
        <div className="alerts-feed-col">
          {filteredAlerts.map((alert) => (
            <div
              key={alert.id}
              className={`ui-card alert-feed-card alert-feed-card--${alert.severity} ${alert.acknowledged ? 'alert-feed-card--ack' : ''}`}
            >
              {/* Card Top Meta */}
              <div className="alert-feed-card__header">
                <div className="alert-meta-left">
                  <span className={`alert-level-badge alert-level-badge--${alert.severity}`}>
                    <span className="level-dot"></span>
                    <span>{alert.severityLabel}</span>
                  </span>
                  <span className="alert-rule-tag">{alert.ruleCode}</span>
                  <span className="alert-line-tag">{alert.line}</span>
                </div>

                <div className="alert-meta-right">
                  {alert.acknowledged ? (
                    <span className="ack-status-tag">
                      <CheckCircle2 size={13} />
                      <span>Resolved</span>
                    </span>
                  ) : (
                    <div className="alert-time-tag">
                      <Clock size={12} />
                      <span>{alert.timeAgo}</span>
                      <span className="alert-time-sep">•</span>
                      <span className="alert-utc">{alert.timestamp}</span>
                    </div>
                  )}
                </div>
              </div>

              {/* Title & Board Ref */}
              <h2 className="alert-feed-title">{alert.title}</h2>
              <div className="alert-board-ref">
                <span>Board Ref: <strong>{alert.boardId}</strong></span>
                {alert.confidence && (
                  <span>• YOLOv8 Confidence <strong className="alert-conf-val">{alert.confidence}</strong></span>
                )}
                {alert.clearedNote && (
                  <span>• {alert.clearedNote}</span>
                )}
              </div>

              {/* Card Body Row (Thumbnail + Trigger/Action) */}
              {alert.severity === 'critical' ? (
                <div className="alert-card-body-row">
                  {/* Thumbnail */}
                  <div className="alert-thumb-box">
                    <img src={alert.thumbnail} alt={alert.title} className="alert-thumb-img" />
                    <span className="alert-thumb-tag">{alert.tagText}</span>
                    <span className="alert-thumb-cam">{alert.camLabel}</span>
                  </div>

                  {/* Trigger & Recommendation */}
                  <div className="alert-trigger-box">
                    <div className="trigger-condition-line">
                      <AlertTriangle size={14} className="trigger-icon" />
                      <span><strong>Trigger Condition:</strong> {alert.triggerText}</span>
                    </div>

                    <div className="recommended-action-box">
                      <strong>Recommended Action:</strong> {alert.recommendedAction}
                    </div>
                  </div>
                </div>
              ) : alert.severity === 'warning' ? (
                <div className="alert-card-body-warning">
                  {alert.triggerRule && (
                    <div className="warning-trigger-line">
                      {alert.triggerRule}
                    </div>
                  )}
                  <p className="warning-desc-text">{alert.detailText}</p>
                </div>
              ) : (
                /* Acknowledged / Resolved format */
                <div className="alert-card-body-resolved">
                  <blockquote className="resolved-quote">
                    {alert.quoteText}
                  </blockquote>
                </div>
              )}

              {/* Bottom Tags and Action Buttons */}
              <div className="alert-feed-card__footer">
                <div className="alert-footer-tags">
                  {alert.metaPills && alert.metaPills.map((pill, idx) => (
                    <span key={idx} className="alert-meta-pill">{pill}</span>
                  ))}
                  {alert.footerTag && (
                    <span className="alert-footer-note">{alert.footerTag}</span>
                  )}
                </div>

                <div className="alert-footer-actions">
                  <button
                    className="alert-btn alert-btn--white"
                    onClick={() => onNavigateToInspection && onNavigateToInspection(alert.boardId)}
                  >
                    <ExternalLink size={13} />
                    <span>View Optical Inspection</span>
                  </button>

                  {!alert.acknowledged && (
                    <button
                      className="alert-btn alert-btn--white"
                      onClick={() => handleAcknowledge(alert.id, alert.title)}
                    >
                      <Check size={13} />
                      <span>Acknowledge</span>
                    </button>
                  )}

                  {alert.actionType === 'dispatch' && !alert.acknowledged && (
                    <button
                      className="alert-btn alert-btn--brown"
                      onClick={() => handleDispatchTech(alert.boardId)}
                    >
                      <Wrench size={13} />
                      <span>Dispatch Rework Tech</span>
                    </button>
                  )}

                  {alert.actionType === 'escalate' && !alert.acknowledged && (
                    <button
                      className="alert-btn alert-btn--red"
                      onClick={() => handleEscalateLead(alert.boardId)}
                    >
                      <Zap size={13} />
                      <span>Escalate to Lead</span>
                    </button>
                  )}
                </div>
              </div>
            </div>
          ))}
        </div>

        {/* Right Column: Triage, Interlocks & Shift Command */}
        <div className="alerts-side-col">
          {/* Panel 1: Severity Ratio & Top Defect Triggers */}
          <div className="ui-card side-panel-card">
            <div className="side-panel-header">
              <h3 className="side-panel-title">Severity Ratio</h3>
              <span className="side-panel-badge">Active 22 Total</span>
            </div>

            {/* Segmented Bar */}
            <div className="severity-bar-track">
              <div className="severity-bar-segment severity-bar-segment--red" style={{ width: '36.4%' }}></div>
              <div className="severity-bar-segment severity-bar-segment--amber" style={{ width: '63.6%' }}></div>
            </div>

            {/* Severity Legend */}
            <div className="severity-legend-row">
              <div className="severity-legend-item">
                <span className="legend-dot-red"></span>
                <span className="legend-name">CRITICAL</span>
                <span className="legend-num">8</span>
                <span className="legend-pct">36.4%</span>
              </div>
              <div className="severity-legend-item">
                <span className="legend-dot-amber"></span>
                <span className="legend-name">WARNINGS</span>
                <span className="legend-num">14</span>
                <span className="legend-pct">63.6%</span>
              </div>
            </div>

            {/* Top Triggers List */}
            <div className="top-triggers-section">
              <span className="top-triggers-title">TOP DEFECT TRIGGERS (24H)</span>
              <div className="top-triggers-list">
                <div className="trigger-row">
                  <div className="trigger-row-left">
                    <span className="trigger-dot trigger-dot--red"></span>
                    <span>Micro Short Circuit</span>
                  </div>
                  <span className="trigger-count">9 occurrences</span>
                </div>
                <div className="trigger-row">
                  <div className="trigger-row-left">
                    <span className="trigger-dot trigger-dot--amber"></span>
                    <span>Spurious Copper Silver</span>
                  </div>
                  <span className="trigger-count">6 occurrences</span>
                </div>
                <div className="trigger-row">
                  <div className="trigger-row-left">
                    <span className="trigger-dot trigger-dot--red"></span>
                    <span>Open Circuit Bus</span>
                  </div>
                  <span className="trigger-count">4 occurrences</span>
                </div>
                <div className="trigger-row">
                  <div className="trigger-row-left">
                    <span className="trigger-dot trigger-dot--amber"></span>
                    <span>Mouse Bite Burr</span>
                  </div>
                  <span className="trigger-count">3 occurrences</span>
                </div>
              </div>
            </div>
          </div>

          {/* Panel 2: Safety Interlocks */}
          <div className="ui-card side-panel-card">
            <div className="side-panel-header">
              <div className="side-panel-title-with-icon">
                <Shield size={16} className="shield-icon" />
                <h3 className="side-panel-title">Safety Interlocks</h3>
              </div>
              <span className="interlocks-active-pill">Active</span>
            </div>

            <div className="interlocks-list">
              {/* Switch 1 */}
              <div className="interlock-item">
                <div className="interlock-text">
                  <span className="interlock-name">Auto-pause SMT Feeder</span>
                  <span className="interlock-sub">Halt conveyor if &ge;5 consecutive shorts detected</span>
                </div>
                <label className="toggle-switch">
                  <input
                    type="checkbox"
                    checked={interlocks.autoPause}
                    onChange={(e) => setInterlocks({ ...interlocks, autoPause: e.target.checked })}
                  />
                  <span className="toggle-slider"></span>
                </label>
              </div>

              {/* Switch 2 */}
              <div className="interlock-item">
                <div className="interlock-text">
                  <span className="interlock-name">Notify Rework Dispatch</span>
                  <span className="interlock-sub">Push immediate buzzer on IPC Class 3 defect</span>
                </div>
                <label className="toggle-switch">
                  <input
                    type="checkbox"
                    checked={interlocks.notifyDispatch}
                    onChange={(e) => setInterlocks({ ...interlocks, notifyDispatch: e.target.checked })}
                  />
                  <span className="toggle-slider"></span>
                </label>
              </div>

              {/* Switch 3 */}
              <div className="interlock-item">
                <div className="interlock-text">
                  <span className="interlock-name">Slack + PagerDuty Bridge</span>
                  <span className="interlock-sub">Channel: <code>#smt-tier1-incidents</code></span>
                </div>
                <label className="toggle-switch">
                  <input
                    type="checkbox"
                    checked={interlocks.slackPager}
                    onChange={(e) => setInterlocks({ ...interlocks, slackPager: e.target.checked })}
                  />
                  <span className="toggle-slider"></span>
                </label>
              </div>
            </div>

            <button
              className="create-trigger-btn"
              onClick={() => showToast('Trigger configuration wizard opened')}
            >
              <Plus size={14} />
              <span>Create Automation Trigger</span>
            </button>
          </div>

          {/* Panel 3: Shift Incident Command */}
          <div className="ui-card side-panel-card">
            <div className="side-panel-header">
              <span className="shift-command-label">SHIFT INCIDENT COMMAND</span>
              <span className="shift-pill">Shift Alpha</span>
            </div>

            <div className="officer-card">
              <div className="officer-avatar-box">
                <span className="officer-initials">SC</span>
                <span className="officer-status-dot"></span>
              </div>
              <div className="officer-info">
                <h4 className="officer-name">Sarah Chen, PE</h4>
                <p className="officer-role">Lead Quality Reliability Engineer</p>
                <p className="officer-contact">Ext: <strong>4082</strong> • Bay: <strong>SMT-03</strong></p>
              </div>
            </div>

            <div className="officer-actions-row">
              <button
                className="officer-btn officer-btn--white"
                onClick={() => showToast('Connecting to Radio Channel Alpha...')}
              >
                <Phone size={13} />
                <span>Direct Radio</span>
              </button>
              <button
                className="officer-btn officer-btn--brown"
                onClick={() => showToast('Shift Lead Sarah Chen paged via PagerDuty')}
              >
                <Radio size={13} />
                <span>Page Shift Lead</span>
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
