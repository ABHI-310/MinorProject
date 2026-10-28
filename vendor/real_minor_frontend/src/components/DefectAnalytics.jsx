import React, { useState } from 'react';
import {
  BarChart3,
  Search,
  Filter,
  Layers,
  Activity,
  AlertTriangle,
  CheckCircle2,
  ExternalLink,
  Download,
  Info,
  Maximize2
} from 'lucide-react';
import './DefectAnalytics.css';

export default function DefectAnalytics({ onSelectDefectType, onBackToOverview }) {
  const [selectedFilter, setSelectedFilter] = useState('all');
  const [activeModalImage, setActiveModalImage] = useState(null);

  const defectClasses = [
    {
      id: 'short',
      name: 'Short (Solder Bridge)',
      severity: 'critical',
      badgeClass: 'badge--critical',
      frequency: '29.0%',
      count: 54,
      totalAnnotations: '4,050',
      description: 'Unintended conductive connection between two or more adjacent copper traces, pins, or solder pads.',
      rootCause: 'Excessive solder paste deposition, stencil aperture wear, or reflow component misalignment.',
      ipcLevel: 'IPC-A-610 Class 3: Critical Defect Non-Conformance',
      image: '/pcb_aoi_inspection.jpg',
      roiImage: '/roi_solder_bridge.jpg',
      sampleFile: 'pred_01_short_01_jpg.rf.543b3693d0c3f26d871a940538ed0ba8.jpg'
    },
    {
      id: 'open_circuit',
      name: 'Open Circuit',
      severity: 'critical',
      badgeClass: 'badge--critical',
      frequency: '17.2%',
      count: 32,
      totalAnnotations: '2,410',
      description: 'Physical discontinuity, break, or void in a copper conductor preventing electrical current flow.',
      rootCause: 'Chemical over-etching, scratch damage during handling, or particulate masking during photoresist exposure.',
      ipcLevel: 'IPC-A-610 Class 3: Immediate Rejection',
      image: '/prediction_samples/pred_01_open_circuit_12_jpg.rf.640be935ceb2f18fbee221d8cf56bad3.jpg',
      roiImage: '/prediction_samples/pred_01_open_circuit_12_jpg.rf.640be935ceb2f18fbee221d8cf56bad3.jpg',
      sampleFile: 'pred_01_open_circuit_12_jpg.rf.640be935ceb2f18fbee221d8cf56bad3.jpg'
    },
    {
      id: 'missing_hole',
      name: 'Missing Hole',
      severity: 'warning',
      badgeClass: 'badge--warning',
      frequency: '20.4%',
      count: 38,
      totalAnnotations: '2,845',
      description: 'Omission of a required drilled via or through-hole on pad geometry.',
      rootCause: 'Broken CNC drill bit during panel processing, skipped NC drill file step, or clogged debris.',
      ipcLevel: 'IPC-A-600 Class 2/3: Non-Conformance',
      image: '/prediction_samples/pred_01_missing_hole_03_jpg.rf.2fe59ac9823cde169d3e2049886eaf6a.jpg',
      roiImage: '/prediction_samples/pred_01_missing_hole_03_jpg.rf.2fe59ac9823cde169d3e2049886eaf6a.jpg',
      sampleFile: 'pred_01_missing_hole_03_jpg.rf.2fe59ac9823cde169d3e2049886eaf6a.jpg'
    },
    {
      id: 'mouse_bite',
      name: 'Mouse Bite',
      severity: 'warning',
      badgeClass: 'badge--warning',
      frequency: '14.0%',
      count: 26,
      totalAnnotations: '1,980',
      description: 'Erosion or irregular bite-shaped edge indentation along trace boundaries reducing nominal track width.',
      rootCause: 'Inconsistent photoresist adhesion, aggressive developing spray, or rough mechanical breakout perforation.',
      ipcLevel: 'IPC-A-600: Acceptable if >75% minimum trace width is maintained',
      image: '/prediction_samples/pred_01_mouse_bite_05_jpg.rf.f4cded1996897893b5e0216f9a365cb8.jpg',
      roiImage: '/prediction_samples/pred_01_mouse_bite_05_jpg.rf.f4cded1996897893b5e0216f9a365cb8.jpg',
      sampleFile: 'pred_01_mouse_bite_05_jpg.rf.f4cded1996897893b5e0216f9a365cb8.jpg'
    },
    {
      id: 'spur',
      name: 'Spur (Burr Outgrowth)',
      severity: 'warning',
      badgeClass: 'badge--warning',
      frequency: '10.8%',
      count: 20,
      totalAnnotations: '1,512',
      description: 'Sharp localized protrusion of conductive copper extending outward from track boundary toward adjacent net.',
      rootCause: 'Micro-pinholes in dry-film photoresist, imperfect developer rinse, or copper burr during drilling.',
      ipcLevel: 'IPC-A-610: Defect if clearance reduced below 0.10mm minimum spacing',
      image: '/roi_copper_spur.jpg',
      roiImage: '/roi_copper_spur.jpg',
      sampleFile: 'pred_01_spur_05_jpg.rf.1ff0881213a4714603198bb630a16247.jpg'
    },
    {
      id: 'spurious_copper',
      name: 'Spurious Copper',
      severity: 'info',
      badgeClass: 'badge--info',
      frequency: '8.6%',
      count: 16,
      totalAnnotations: '1,151',
      description: 'Isolated island of unetched residual copper speckle left on the dielectric base substrate.',
      rootCause: 'Uneven etchant fluid circulation, particulate contamination during lamination, or resist residue.',
      ipcLevel: 'IPC-A-600: Permissible if electrically isolated and clear of pads',
      image: '/prediction_samples/pred_01_spurious_copper_02_jpg.rf.0a382538d512ca6797cbf5791185a974.jpg',
      roiImage: '/prediction_samples/pred_01_spurious_copper_02_jpg.rf.0a382538d512ca6797cbf5791185a974.jpg',
      sampleFile: 'pred_01_spurious_copper_02_jpg.rf.0a382538d512ca6797cbf5791185a974.jpg'
    }
  ];

  const filteredDefects = defectClasses.filter((d) => {
    if (selectedFilter === 'all') return true;
    return d.severity === selectedFilter;
  });

  return (
    <div className="analytics-view">
      {/* Image Zoom Modal */}
      {activeModalImage && (
        <div className="analytics-modal-backdrop" onClick={() => setActiveModalImage(null)}>
          <div className="analytics-modal-box" onClick={(e) => e.stopPropagation()}>
            <div className="analytics-modal-header">
              <span className="analytics-modal-title">{activeModalImage.title}</span>
              <button className="analytics-modal-close" onClick={() => setActiveModalImage(null)}>✕</button>
            </div>
            <div className="analytics-modal-img-wrap">
              <img src={activeModalImage.src} alt={activeModalImage.title} className="analytics-modal-img" />
            </div>
          </div>
        </div>
      )}

      {/* Top Breadcrumb & Header */}
      <div className="analytics-topbar">
        <div className="analytics-topbar__left">
          <nav className="analytics-breadcrumbs">
            <button className="breadcrumb-link" onClick={onBackToOverview}>Overview</button>
            <span className="breadcrumb-sep">&gt;</span>
            <span className="breadcrumb-current">Defect Analytics & Inspection Gallery</span>
          </nav>
          <h1 className="analytics-title">Defect Analytics & Multi-Class Diagnostics</h1>
          <p className="analytics-subtitle">
            Comprehensive defect characterization across 6 standard IPC anomaly classes, model confusion matrices, and training curves.
          </p>
        </div>

        <div className="analytics-topbar__right">
          <button className="topbar-btn topbar-btn--white" onClick={() => window.print()}>
            <Download size={14} />
            <span>Export Analytics Report</span>
          </button>
        </div>
      </div>

      {/* Top 4 Summary Cards */}
      <div className="analytics-metrics-grid">
        <div className="ui-card analytics-stat-card">
          <span className="stat-label">TOTAL ANNOTATIONS</span>
          <div className="stat-val-row">
            <span className="stat-val">13,948</span>
            <span className="stat-unit">boxes</span>
          </div>
          <span className="stat-sub">Across 3,326 curated images</span>
        </div>

        <div className="ui-card analytics-stat-card">
          <span className="stat-label">STANDARDIZED CLASSES</span>
          <div className="stat-val-row">
            <span className="stat-val">6</span>
            <span className="stat-unit">categories</span>
          </div>
          <span className="stat-sub">IPC-A-600/610 taxonomy</span>
        </div>

        <div className="ui-card analytics-stat-card">
          <span className="stat-label">BASELINE PRECISION</span>
          <div className="stat-val-row">
            <span className="stat-val">28.9%</span>
            <span className="stat-unit stat-unit--amber">Epoch 50</span>
          </div>
          <span className="stat-sub">YOLOv8n-640 Baseline</span>
        </div>

        <div className="ui-card analytics-stat-card">
          <span className="stat-label">FINE-TUNED TARGET</span>
          <div className="stat-val-row">
            <span className="stat-val stat-val--green">99.4%</span>
            <span className="stat-unit stat-unit--green">F1 Score</span>
          </div>
          <span className="stat-sub">Domain-adapted YOLOv8m</span>
        </div>
      </div>

      {/* Section 1: 6 Defect Types Visual Inspection Gallery */}
      <div className="analytics-section">
        <div className="analytics-section__header">
          <div className="section-title-wrap">
            <Layers size={18} className="section-icon" />
            <h2 className="section-title">6-Class Optical Defect Gallery</h2>
            <span className="section-badge">YOLOv8 Real Inference Samples</span>
          </div>

          <div className="gallery-filter-tabs">
            {['all', 'critical', 'warning', 'info'].map((f) => (
              <button
                key={f}
                className={`gallery-filter-btn ${selectedFilter === f ? 'gallery-filter-btn--active' : ''}`}
                onClick={() => setSelectedFilter(f)}
              >
                {f.charAt(0).toUpperCase() + f.slice(1)}
              </button>
            ))}
          </div>
        </div>

        <div className="defect-gallery-grid">
          {filteredDefects.map((defect) => (
            <div key={defect.id} className="ui-card defect-gallery-card">
              {/* Image Frame */}
              <div
                className="defect-gallery-card__img-frame"
                onClick={() => setActiveModalImage({ src: defect.image, title: `${defect.name} - Optical Sample` })}
                title="Click to view full image"
              >
                <img src={defect.image} alt={defect.name} className="defect-gallery-img" />
                <div className="gallery-overlay-badge">
                  <span className={`badge ${defect.badgeClass}`}>
                    {defect.severity.toUpperCase()}
                  </span>
                  <span className="gallery-share-pill">{defect.frequency} share</span>
                </div>
                <button className="gallery-expand-btn">
                  <Maximize2 size={13} />
                </button>
              </div>

              {/* Card Body */}
              <div className="defect-gallery-card__body">
                <div className="defect-card-header">
                  <h3 className="defect-card-name">{defect.name}</h3>
                  <span className="defect-card-count">
                    <strong>{defect.count}</strong> / {defect.totalAnnotations} total
                  </span>
                </div>

                <p className="defect-card-desc">{defect.description}</p>

                <div className="defect-card-meta">
                  <div className="meta-line">
                    <span className="meta-key">Root Cause:</span>
                    <span className="meta-val">{defect.rootCause}</span>
                  </div>
                  <div className="meta-line">
                    <span className="meta-key">IPC Standard:</span>
                    <span className="meta-val meta-val--ipc">{defect.ipcLevel}</span>
                  </div>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Section 2: Model Evaluation Artifacts (Confusion Matrix & Loss Curves) */}
      <div className="analytics-section">
        <div className="analytics-section__header">
          <div className="section-title-wrap">
            <Activity size={18} className="section-icon" />
            <h2 className="section-title">Model Diagnostics & Evaluation Artifacts</h2>
            <span className="section-badge">Ultra-High Precision Reports</span>
          </div>
        </div>

        <div className="artifacts-grid">
          {/* Artifact 1: Confusion Matrix */}
          <div className="ui-card artifact-card">
            <div className="artifact-card__header">
              <div className="artifact-title-box">
                <h3 className="artifact-title">Multi-Class Confusion Matrix</h3>
                <span className="artifact-sub">Normalized prediction vs ground truth matrix</span>
              </div>
              <button
                className="artifact-zoom-btn"
                onClick={() => setActiveModalImage({ src: '/confusion_matrix.png', title: 'PCB Defect Detection Confusion Matrix' })}
              >
                <Maximize2 size={14} />
                <span>Enlarge</span>
              </button>
            </div>

            <div
              className="artifact-img-frame"
              onClick={() => setActiveModalImage({ src: '/confusion_matrix.png', title: 'PCB Defect Detection Confusion Matrix' })}
            >
              <img
                src="/confusion_matrix.png"
                alt="Confusion Matrix"
                className="artifact-img"
              />
            </div>

            <div className="artifact-footer">
              <span className="artifact-note">
                Diagonal elements represent correct positive classifications across the 6 defect categories.
              </span>
            </div>
          </div>

          {/* Artifact 2: Training Loss Curves */}
          <div className="ui-card artifact-card">
            <div className="artifact-card__header">
              <div className="artifact-title-box">
                <h3 className="artifact-title">Training Loss & Convergence Curves</h3>
                <span className="artifact-sub">Box Loss, Class Loss, DFL Loss & mAP over epochs</span>
              </div>
              <button
                className="artifact-zoom-btn"
                onClick={() => setActiveModalImage({ src: '/training_results.png', title: 'YOLOv8 Training Convergence Curves' })}
              >
                <Maximize2 size={14} />
                <span>Enlarge</span>
              </button>
            </div>

            <div
              className="artifact-img-frame"
              onClick={() => setActiveModalImage({ src: '/training_results.png', title: 'YOLOv8 Training Convergence Curves' })}
            >
              <img
                src="/training_results.png"
                alt="Training Loss Results"
                className="artifact-img"
              />
            </div>

            <div className="artifact-footer">
              <span className="artifact-note">
                Tracks bounding box regression, distribution focal loss, and validation mAP@0.5:0.95 trajectory.
              </span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
