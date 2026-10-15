import React from 'react';
import { X, CheckCircle, AlertTriangle, Download, Layers, ShieldCheck } from 'lucide-react';
import './InspectionModal.css';

export default function InspectionModal({ inspection, onClose }) {
  if (!inspection) return null;

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div className="modal-content" onClick={(e) => e.stopPropagation()}>
        {/* Header */}
        <div className="modal-header">
          <div className="modal-title-group">
            <div className="modal-id-row">
              <span className="modal-id">{inspection.id}</span>
              <span className={`status-badge status-badge--${inspection.statusType}`}>
                {inspection.status}
              </span>
            </div>
            <span className="modal-subtitle">
              High-Precision AOI Optical & YOLOv8 Inference Report
            </span>
          </div>

          <button className="modal-close-btn" onClick={onClose}>
            <X size={18} />
          </button>
        </div>

        {/* Body */}
        <div className="modal-body">
          {/* Left: Image Viewer */}
          <div className="modal-image-col">
            <div className="modal-image-frame">
              <img
                src={inspection.thumbnail}
                alt={inspection.id}
                className="modal-image"
              />
              <div className="modal-image-overlay">
                <span className="overlay-tag">YOLOv8 Bounding Overlay</span>
                <span className="overlay-res">640 × 640 Native Res</span>
              </div>
            </div>
            <div className="modal-operator-row">
              <span>Station: <strong>{inspection.operator || 'Auto-AOI-01'}</strong></span>
              <span>Lot: <strong>{inspection.lot || 'Lot B4-8842'}</strong></span>
              <span>Timestamp: <strong>{inspection.timestamp} UTC</strong></span>
            </div>
          </div>

          {/* Right: Inspection Diagnostics */}
          <div className="modal-info-col">
            <div className="modal-section">
              <span className="modal-section-title">DEFECT CLASSIFICATION</span>
              <div className="modal-defect-badge" style={{ backgroundColor: inspection.badgeBg, color: inspection.color }}>
                <span className="modal-defect-dot" style={{ backgroundColor: inspection.color }}></span>
                <span className="modal-defect-name">{inspection.defectType}</span>
                <span className="modal-defect-conf">({inspection.confidence}%)</span>
              </div>
            </div>

            <div className="modal-section">
              <span className="modal-section-title">ANOMALY SUMMARY</span>
              <p className="modal-defect-desc">
                {inspection.details}
              </p>
            </div>

            <div className="modal-section">
              <span className="modal-section-title">LAYER INSPECTION</span>
              <div className="modal-layer-box">
                <Layers size={14} className="layer-icon" />
                <span>{inspection.layer || 'Layer 2 Signal Traces'}</span>
              </div>
            </div>

            <div className="modal-section">
              <span className="modal-section-title">ACTION & WORKFLOW</span>
              <div className="modal-actions-grid">
                <button className="modal-btn modal-btn--approve" onClick={onClose}>
                  <CheckCircle size={14} />
                  <span>Approve Board</span>
                </button>
                <button className="modal-btn modal-btn--rework" onClick={onClose}>
                  <AlertTriangle size={14} />
                  <span>Flag for Rework</span>
                </button>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
