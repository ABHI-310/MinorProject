import React from 'react';
import { usePipeline } from '../data/PipelineContext';
import './RuntimeBanner.css';

export default function RuntimeBanner() {
  const { systemMetrics } = usePipeline();
  return (
    <div className="runtime-banner">
      <div className="runtime-banner__left">
        <span className="runtime-banner__dot"></span>
        <span className="runtime-banner__tag">RUNTIME PROFILE</span>
        <span className="runtime-banner__divider">•</span>
        <span className="runtime-banner__desc">
          <strong>{systemMetrics.systemStatus}</strong> — Real inference and saved inspection history
        </span>
      </div>

      <div className="runtime-banner__right">
        <span className="runtime-banner__model-badge">
          Model: {systemMetrics.modelName}
        </span>
        <span className="runtime-banner__f1-badge">
          <span className="f1-dot"></span>
          {systemMetrics.f1Score}
        </span>
      </div>
    </div>
  );
}
