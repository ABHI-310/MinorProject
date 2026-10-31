import React from 'react';
import {
  Cpu,
  Target,
  FileCheck2,
  AlertTriangle,
  TrendingUp,
  TrendingDown
} from 'lucide-react';
import { usePipeline } from '../data/PipelineContext';
import './MetricsPanel.css';

export default function MetricsPanel({ onActionReq }) {
  const { systemMetrics } = usePipeline();
  return (
    <div className="metrics-grid">
      {/* Card 1: PCBs Inspected */}
      <div className="ui-card metric-card">
        <div className="metric-card__header">
          <span className="metric-card__title">INSPECTIONS SAVED</span>
          <div className="metric-card__icon-box metric-card__icon-box--orange">
            <Cpu size={16} />
          </div>
        </div>
        <div className="metric-card__value-row">
          <span className="metric-card__main-val">{systemMetrics.inspectedFormatted}</span>
          <span className="metric-card__unit">inspections</span>
        </div>
        <div className="metric-card__footer">
          <span className="metric-card__footer-text" title={systemMetrics.inspectedSubtitle}>
            {systemMetrics.inspectedSubtitle}
          </span>
          <span className="metric-pill metric-pill--green">
            <TrendingUp size={12} />
            <span>{systemMetrics.inspectedTrend}</span>
          </span>
        </div>
      </div>

      {/* Card 2: Defects Detected */}
      <div className="ui-card metric-card">
        <div className="metric-card__header">
          <span className="metric-card__title">DEFECTS DETECTED</span>
          <div className="metric-card__icon-box metric-card__icon-box--blue">
            <Target size={16} />
          </div>
        </div>
        <div className="metric-card__value-row">
          <span className="metric-card__main-val">{systemMetrics.defectsDetected}</span>
          <span className="metric-card__unit">instances</span>
        </div>
        <div className="metric-card__footer">
          <span className="metric-card__footer-text" title={systemMetrics.defectsSubtitle}>
            {systemMetrics.defectsSubtitle}
          </span>
          <span className="metric-pill metric-pill--green">
            <TrendingDown size={12} />
            <span>{systemMetrics.defectsRate}</span>
          </span>
        </div>
      </div>

      {/* Card 3: Defective Boards */}
      <div className="ui-card metric-card">
        <div className="metric-card__header">
          <span className="metric-card__title">INSPECTIONS WITH DEFECTS</span>
          <div className="metric-card__icon-box metric-card__icon-box--amber">
            <FileCheck2 size={16} />
          </div>
        </div>
        <div className="metric-card__value-row">
          <span className="metric-card__main-val">{systemMetrics.defectiveBoards}</span>
          <span className="metric-card__unit">inspections</span>
        </div>
        <div className="metric-card__footer">
          <span className="metric-card__footer-text" title={systemMetrics.defectsPerBoard}>
            {systemMetrics.defectsPerBoard}
          </span>
          <span className="metric-pill metric-pill--purple">
            <span>{systemMetrics.defectYield}</span>
          </span>
        </div>
      </div>

      {/* Card 4: Critical Alerts */}
      <div className="ui-card metric-card metric-card--alert">
        <div className="metric-card__header">
          <span className="metric-card__title">CRITICAL DETECTIONS</span>
          <div className="metric-card__icon-box metric-card__icon-box--red">
            <AlertTriangle size={16} />
            <span className="alert-ping-dot"></span>
          </div>
        </div>
        <div className="metric-card__value-row">
          <span className="metric-card__main-val metric-card__main-val--red">
            {systemMetrics.criticalFormatted}
          </span>
          <span className="metric-card__unit">findings</span>
        </div>
        <div className="metric-card__footer">
          <span className="metric-card__footer-text" title={systemMetrics.criticalSubtitle}>
            {systemMetrics.criticalSubtitle}
          </span>
          <button
            className="metric-btn-action"
            onClick={onActionReq}
            title="View Critical Alerts"
          >
            ACTION REQ.
          </button>
        </div>
      </div>
    </div>
  );
}
