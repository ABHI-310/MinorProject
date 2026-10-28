import React from 'react';
import { systemMetrics } from '../data/mockDetections';
import './RuntimeBanner.css';

export default function RuntimeBanner() {
  return (
    <div className="runtime-banner">
      <div className="runtime-banner__left">
        <span className="runtime-banner__dot"></span>
        <span className="runtime-banner__tag">RUNTIME PROFILE</span>
        <span className="runtime-banner__divider">•</span>
        <span className="runtime-banner__desc">
          <strong>Demo Data Mode</strong> — College Prototype Evaluation Environment (FastAPI Mock Server Active)
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
