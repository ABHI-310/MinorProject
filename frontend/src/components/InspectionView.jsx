import React, { useState } from 'react';
import {
  ArrowLeft,
  RotateCw,
  Download,
  Check,
  CheckCircle2,
  Layers,
  Crosshair,
  ZoomIn,
  ZoomOut,
  Maximize2,
  Eye,
  Sliders,
  AlertTriangle,
  Printer,
  GraduationCap,
  Sparkles,
  Flame,
  Scan
} from 'lucide-react';
import './InspectionView.css';

export default function InspectionView({ inspection, onBackToOverview }) {
  // Mode state
  const [viewMode, setViewMode] = useState('ai_overlay'); // 'ai_overlay' | 'raw' | 'split'
  const [zoomLevel, setZoomLevel] = useState(100);
  const [showBoxes, setShowBoxes] = useState(true);
  const [showHeatmap, setShowHeatmap] = useState(false);
  const [activeDefectId, setActiveDefectId] = useState('#01');
  const [disposition, setDisposition] = useState('rework');
  const [notes, setNotes] = useState('');
  const [isVerifying, setIsVerifying] = useState(false);
  const [isVerified, setIsVerified] = useState(false);
  const [isRerunning, setIsRerunning] = useState(false);
  const [toastMessage, setToastMessage] = useState(null);

  const boardId = inspection?.id || '#PCB-2025-0842';

  const showToast = (msg) => {
    setToastMessage(msg);
    setTimeout(() => setToastMessage(null), 3000);
  };

  const handleRerunInference = () => {
    setIsRerunning(true);
    setTimeout(() => {
      setIsRerunning(false);
      showToast('Inference re-evaluated: 2 defects confirmed (42.1ms)');
    }, 900);
  };

  const handleSignOff = () => {
    showToast(`Board ${boardId} successfully signed off & routed to ${disposition.toUpperCase()}`);
    setIsVerified(true);
  };

  return (
    <div className="inspection-view">
      {/* Toast Notification */}
      {toastMessage && (
        <div className="inspection-toast">
          <CheckCircle2 size={16} />
          <span>{toastMessage}</span>
        </div>
      )}

      {/* Top Breadcrumb & Action Bar */}
      <div className="inspection-topbar">
        <div className="inspection-topbar__left">
          <nav className="inspection-breadcrumbs">
            <button className="breadcrumb-link" onClick={onBackToOverview}>Overview</button>
            <span className="breadcrumb-sep">&gt;</span>
            <span className="breadcrumb-current">Inspections</span>
            <span className="breadcrumb-sep">&gt;</span>
            <span className="breadcrumb-id">{boardId}</span>
          </nav>

          <div className="inspection-title-wrap">
            <h1 className="inspection-title">Inspection Analysis: {boardId}</h1>
          </div>

          <div className="inspection-anomaly-pill">
            <span className="anomaly-dot"></span>
            <span>CRITICAL ANOMALY: SHORT DETECTED</span>
          </div>
        </div>

        {/* Action Buttons */}
        <div className="inspection-topbar__right">
          <button className="topbar-btn topbar-btn--white" onClick={onBackToOverview}>
            <ArrowLeft size={14} />
            <span>Back to Overview</span>
          </button>

          <button
            className={`topbar-btn topbar-btn--blue ${isRerunning ? 'btn-spinning' : ''}`}
            onClick={handleRerunInference}
            disabled={isRerunning}
          >
            <RotateCw size={14} className={isRerunning ? 'spin-icon' : ''} />
            <span>{isRerunning ? 'Inferencing...' : 'Rerun Inference'}</span>
          </button>

          <button
            className="topbar-btn topbar-btn--white"
            onClick={() => window.print()}
            title="Export Inspection Report"
          >
            <Download size={14} />
            <span>Export Report</span>
          </button>

          <button
            className={`topbar-btn topbar-btn--brown ${isVerified ? 'topbar-btn--verified' : ''}`}
            onClick={() => {
              setIsVerified(!isVerified);
              showToast(isVerified ? 'Board marked as unverified' : 'Board marked as verified');
            }}
          >
            <Check size={14} />
            <span>{isVerified ? 'Verified' : 'Mark as Verified'}</span>
          </button>
        </div>
      </div>

      {/* Main 2-Column Inspection Grid */}
      <div className="inspection-grid">
        {/* Left Column: Image Viewer & ROI Crops */}
        <div className="inspection-grid__left">
          {/* AOI Core Optical Viewer Card */}
          <div className="ui-card aoi-viewer-card">
            {/* Dark Card Header */}
            <div className="aoi-viewer-header">
              <div className="aoi-viewer-header__left">
                <Scan size={16} className="aoi-icon" />
                <span className="aoi-title">MULTI-SPECTRUM AOI CORE</span>
                <span className="aoi-sensor-badge">12MP Sony IMX477</span>
              </div>

              <div className="aoi-mode-toggles">
                <button
                  className={`aoi-mode-btn ${viewMode === 'ai_overlay' ? 'aoi-mode-btn--active' : ''}`}
                  onClick={() => setViewMode('ai_overlay')}
                >
                  AI Overlay
                </button>
                <button
                  className={`aoi-mode-btn ${viewMode === 'raw' ? 'aoi-mode-btn--active' : ''}`}
                  onClick={() => setViewMode('raw')}
                >
                  Raw Optical
                </button>
                <button
                  className={`aoi-mode-btn ${viewMode === 'split' ? 'aoi-mode-btn--active' : ''}`}
                  onClick={() => setViewMode('split')}
                >
                  Split Compare
                </button>
              </div>
            </div>

            {/* Viewer Stage */}
            <div className="aoi-viewer-stage">
              <div
                className="aoi-image-container"
                style={{ transform: `scale(${zoomLevel / 100})` }}
              >
                {/* PCB Image */}
                <img
                  src={boardId === '#PCB-2025-0842' ? '/pcb_aoi_inspection.jpg' : (inspection?.thumbnail || '/pcb_aoi_inspection.jpg')}
                  alt="High-resolution PCB AOI Inspection"
                  className={`aoi-main-image ${viewMode === 'raw' ? 'aoi-main-image--raw' : ''}`}
                />

                {/* Simulated Heatmap Overlay */}
                {showHeatmap && (
                  <div className="aoi-heatmap-overlay"></div>
                )}

                {/* Bounding Boxes (Only visible when AI overlay and showBoxes are on) */}
                {viewMode !== 'raw' && showBoxes && (
                  <>
                    {/* Bounding Box 1: Short */}
                    <div
                      className={`aoi-bbox aoi-bbox--short ${activeDefectId === '#01' ? 'aoi-bbox--active' : ''}`}
                      style={{ top: '44%', left: '46.5%', width: '10.5%', height: '14%' }}
                      onClick={() => setActiveDefectId('#01')}
                    >
                      <div className="bbox-pill bbox-pill--red">
                        <span className="bbox-dot"></span>
                        <span>short 98.2%</span>
                      </div>
                      <div className="bbox-sublabel">#D01 • Pin 14-15</div>
                      <div className="bbox-corner bbox-corner--tl"></div>
                      <div className="bbox-corner bbox-corner--tr"></div>
                      <div className="bbox-corner bbox-corner--bl"></div>
                      <div className="bbox-corner bbox-corner--br"></div>
                    </div>

                    {/* Bounding Box 2: Spur */}
                    <div
                      className={`aoi-bbox aoi-bbox--spur ${activeDefectId === '#02' ? 'aoi-bbox--active' : ''}`}
                      style={{ top: '58%', left: '63%', width: '8.5%', height: '11%' }}
                      onClick={() => setActiveDefectId('#02')}
                    >
                      <div className="bbox-pill bbox-pill--spur">
                        <span className="bbox-dot"></span>
                        <span>spur 89.4%</span>
                      </div>
                      <div className="bbox-sublabel">#D02</div>
                      <div className="bbox-corner bbox-corner--tl"></div>
                      <div className="bbox-corner bbox-corner--tr"></div>
                      <div className="bbox-corner bbox-corner--bl"></div>
                      <div className="bbox-corner bbox-corner--br"></div>
                    </div>
                  </>
                )}
              </div>

              {/* Floating Bottom Toolbar */}
              <div className="aoi-floating-toolbar">
                <div className="toolbar-model-chip">
                  <span className="status-dot status-dot--online"></span>
                  <span className="toolbar-model-name">YOLOv8-PCB-Nano</span>
                </div>
                <div className="toolbar-divider"></div>
                <div className="toolbar-latency">
                  Latency: <strong>42.1ms</strong>
                </div>
                <div className="toolbar-divider"></div>
                {/* Zoom Controls */}
                <div className="toolbar-zoom-group">
                  <button
                    className="toolbar-btn"
                    onClick={() => setZoomLevel((z) => Math.max(60, z - 15))}
                    title="Zoom Out"
                  >
                    <ZoomOut size={13} />
                  </button>
                  <span className="toolbar-zoom-text">{zoomLevel}%</span>
                  <button
                    className="toolbar-btn"
                    onClick={() => setZoomLevel((z) => Math.min(200, z + 15))}
                    title="Zoom In"
                  >
                    <ZoomIn size={13} />
                  </button>
                </div>
                <div className="toolbar-divider"></div>
                {/* Boxes Toggle */}
                <button
                  className={`toolbar-btn-toggle ${showBoxes ? 'toolbar-btn-toggle--active' : ''}`}
                  onClick={() => setShowBoxes(!showBoxes)}
                  title="Toggle Bounding Boxes"
                >
                  <Crosshair size={13} />
                  <span>Boxes</span>
                </button>
                {/* Heatmap Toggle */}
                <button
                  className={`toolbar-btn-toggle ${showHeatmap ? 'toolbar-btn-toggle--active' : ''}`}
                  onClick={() => setShowHeatmap(!showHeatmap)}
                  title="Toggle Activation Heatmap"
                >
                  <Flame size={13} />
                  <span>Heatmap</span>
                </button>
                <div className="toolbar-divider"></div>
                <button
                  className="toolbar-btn"
                  onClick={() => setZoomLevel(100)}
                  title="Reset Zoom / Full View"
                >
                  <Maximize2 size={13} />
                </button>
              </div>
            </div>

            {/* Viewer Bottom Info */}
            <div className="aoi-viewer-meta">
              <div className="aoi-meta-row">
                <span>Pixel Pitch: <strong>3.45 µm/px</strong></span>
                <span>Lens: <strong>50mm Macro f/2.8</strong></span>
                <span>Illumination: <strong>Direct Coaxial Polarized</strong></span>
              </div>
              <div className="aoi-coords-line">
                <Crosshair size={12} />
                <span>Coordinates Normalized [0..1000]</span>
              </div>
            </div>
          </div>

          {/* Localized ROI Defect Crops Card */}
          <div className="ui-card roi-crops-card">
            <div className="roi-crops-header">
              <div className="roi-crops-title-row">
                <Layers size={16} className="roi-icon" />
                <h3 className="roi-crops-title">Localized ROI Defect Crops</h3>
              </div>
              <span className="roi-crops-badge">2 Regions of Interest Flagged</span>
            </div>

            <div className="roi-crops-grid">
              {/* ROI Card 1: Short */}
              <div
                className={`roi-card ${activeDefectId === '#01' ? 'roi-card--active' : ''}`}
                onClick={() => setActiveDefectId('#01')}
              >
                <div className="roi-thumb-frame">
                  <img
                    src="/roi_solder_bridge.jpg"
                    alt="Short defect ROI"
                    className="roi-thumb-img"
                  />
                  <span className="roi-tag-badge roi-tag-badge--red">#D01</span>
                </div>
                <div className="roi-content">
                  <div className="roi-title-line">
                    <span className="roi-defect-name roi-defect-name--red">SHORT</span>
                    <span className="roi-conf-badge roi-conf-badge--red">98.2%</span>
                  </div>
                  <p className="roi-desc">Solder bridge between pin 14-15...</p>
                  <div className="roi-geometry-line">
                    ROI: [412, 184, 68, 42] • 2,856 px²
                  </div>
                </div>
              </div>

              {/* ROI Card 2: Spur */}
              <div
                className={`roi-card ${activeDefectId === '#02' ? 'roi-card--active' : ''}`}
                onClick={() => setActiveDefectId('#02')}
              >
                <div className="roi-thumb-frame">
                  <img
                    src="/roi_copper_spur.jpg"
                    alt="Spur defect ROI"
                    className="roi-thumb-img"
                  />
                  <span className="roi-tag-badge roi-tag-badge--amber">#D02</span>
                </div>
                <div className="roi-content">
                  <div className="roi-title-line">
                    <span className="roi-defect-name roi-defect-name--amber">SPUR</span>
                    <span className="roi-conf-badge roi-conf-badge--amber">89.4%</span>
                  </div>
                  <p className="roi-desc">Minor trace edge copper outgrowth...</p>
                  <div className="roi-geometry-line">
                    ROI: [620, 395, 32, 28] • 896 px²
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Right Column: Telemetry, Geometry, SOP, and Disclaimer */}
        <div className="inspection-grid__right">
          {/* Card 1: Detection Telemetry */}
          <div className="ui-card telemetry-card">
            <div className="telemetry-card__header">
              <div className="telemetry-card__title-row">
                <Crosshair size={16} className="telemetry-icon" />
                <h3 className="telemetry-card__title">Detection Telemetry</h3>
              </div>
              <span className="telemetry-action-badge">Action Required</span>
            </div>

            {/* 2 Top Metric Boxes */}
            <div className="telemetry-metrics-grid">
              <div className="telemetry-box">
                <span className="telemetry-box__label">DEFECT COUNT</span>
                <div className="telemetry-box__val-row">
                  <span className="telemetry-num telemetry-num--red">2</span>
                  <span className="telemetry-unit telemetry-unit--red">Anomalies</span>
                </div>
                <span className="telemetry-box__sub">1 Critical • 1 Non-fatal</span>
              </div>

              <div className="telemetry-box">
                <span className="telemetry-box__label">PEAK CONFIDENCE</span>
                <div className="telemetry-box__val-row">
                  <span className="telemetry-num">98.2%</span>
                  <span className="telemetry-unit telemetry-unit--green">High</span>
                </div>
                <span className="telemetry-box__sub">YOLOv8 Detection Core</span>
              </div>
            </div>

            {/* Telemetry Metadata Table */}
            <div className="telemetry-meta-list">
              <div className="telemetry-meta-item">
                <span className="telemetry-meta-key">Production Line / Station:</span>
                <span className="telemetry-meta-val">SMT Line 01 • Station Alpha</span>
              </div>
              <div className="telemetry-meta-item">
                <span className="telemetry-meta-key">Board Serial Hash:</span>
                <span className="telemetry-meta-val telemetry-meta-val--mono">0x842A_FF910C2</span>
              </div>
              <div className="telemetry-meta-item">
                <span className="telemetry-meta-key">Timestamp (UTC):</span>
                <span className="telemetry-meta-val telemetry-meta-val--mono">2025-05-20 14:28:12 UTC</span>
              </div>
              <div className="telemetry-meta-item">
                <span className="telemetry-meta-key">IPC Classification:</span>
                <span className="telemetry-meta-val telemetry-meta-val--critical">Class 3 Defect Non-Conformance</span>
              </div>
            </div>
          </div>

          {/* Card 2: Bounding Box Geometry */}
          <div className="ui-card geometry-card">
            <div className="geometry-card__header">
              <div className="geometry-card__title-row">
                <Crosshair size={16} className="geometry-icon" />
                <h3 className="geometry-card__title">Bounding Box Geometry</h3>
              </div>
              <span className="geometry-schema-badge">JSON Schema v1.2</span>
            </div>

            <table className="geometry-table">
              <thead>
                <tr>
                  <th>ID</th>
                  <th>CLASS</th>
                  <th>BBOX [X,Y,W,H]</th>
                  <th>AREA</th>
                  <th>CONFIDENCE</th>
                </tr>
              </thead>
              <tbody>
                <tr className={activeDefectId === '#01' ? 'geometry-row--active' : ''} onClick={() => setActiveDefectId('#01')}>
                  <td className="geom-mono">#01</td>
                  <td>
                    <span className="geom-class-pill geom-class-pill--red">
                      <span className="geom-dot geom-dot--red"></span>
                      <span>short</span>
                    </span>
                  </td>
                  <td className="geom-mono">[412, 184, 68, 42]</td>
                  <td className="geom-mono">2,856 px²</td>
                  <td className="geom-mono geom-conf--red">98.2%</td>
                </tr>
                <tr className={activeDefectId === '#02' ? 'geometry-row--active' : ''} onClick={() => setActiveDefectId('#02')}>
                  <td className="geom-mono">#02</td>
                  <td>
                    <span className="geom-class-pill geom-class-pill--amber">
                      <span className="geom-dot geom-dot--amber"></span>
                      <span>spur</span>
                    </span>
                  </td>
                  <td className="geom-mono">[620, 395, 32, 28]</td>
                  <td className="geom-mono">896 px²</td>
                  <td className="geom-mono">89.4%</td>
                </tr>
              </tbody>
            </table>
          </div>

          {/* Card 3: Standard Operating Procedure (SOP) */}
          <div className="ui-card sop-card">
            <div className="sop-card__header">
              <div className="sop-card__title-row">
                <CheckCircle2 size={16} className="sop-icon" />
                <h3 className="sop-card__title">Standard Operating Procedure</h3>
              </div>
              <span className="sop-code-badge">SOP-ENG-08</span>
            </div>

            {/* Amber Operator Recommendation Box */}
            <div className="sop-recommendation-box">
              <div className="sop-rec-header">
                <AlertTriangle size={14} className="sop-alert-icon" />
                <span className="sop-rec-title">Recommended Operator Action</span>
              </div>
              <p className="sop-rec-text">
                Route PCB to <strong>Rework Station #4</strong>. Perform laser desoldering or manual solder braid wick cleanup across pin 14-15 bridge. Recalibrate stencil aperture if bridging recurs in batch.
              </p>
            </div>

            {/* Board Disposition Radio Group */}
            <div className="sop-field">
              <label className="sop-field-label">BOARD DISPOSITION</label>
              <div className="sop-disposition-radios">
                {['rework', 'scrap', 'override'].map((mode) => (
                  <label key={mode} className={`disposition-radio-label ${disposition === mode ? 'disposition-radio-label--active' : ''}`}>
                    <input
                      type="radio"
                      name="disposition"
                      value={mode}
                      checked={disposition === mode}
                      onChange={(e) => setDisposition(e.target.value)}
                    />
                    <span className="radio-custom"></span>
                    <span className="radio-text">{mode.charAt(0).toUpperCase() + mode.slice(1)}</span>
                  </label>
                ))}
              </div>
            </div>

            {/* Notes Textarea */}
            <div className="sop-field">
              <label className="sop-field-label">QUALITY INSPECTOR NOTES</label>
              <textarea
                className="sop-textarea"
                rows={2}
                placeholder="e.g., Confirmed solder bridge. Forwarding tray to rework tech Dave M."
                value={notes}
                onChange={(e) => setNotes(e.target.value)}
              ></textarea>
            </div>

            {/* Form Actions */}
            <div className="sop-actions-row">
              <button className="sop-submit-btn" onClick={handleSignOff}>
                <Check size={15} />
                <span>Sign-off & Route Board</span>
              </button>
              <button
                className="sop-print-btn"
                onClick={() => window.print()}
                title="Print Routing Tag"
              >
                <Printer size={15} />
              </button>
            </div>
          </div>

          {/* Card 4: Prototype Disclaimer */}
          <div className="disclaimer-card">
            <div className="disclaimer-header">
              <GraduationCap size={15} className="disclaimer-icon" />
              <span className="disclaimer-title">PROTOTYPE EVALUATION DISCLAIMER</span>
            </div>
            <p className="disclaimer-text">
              Demonstration telemetry for university evaluation prototype. Bounding boxes, inference metrics, and thermal maps are illustrative mockups rendered for UI and system architecture validation.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
