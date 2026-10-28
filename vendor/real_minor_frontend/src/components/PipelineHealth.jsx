import React from 'react';
import { CheckCircle2 } from 'lucide-react';
import { pipelineHealth } from '../data/mockDetections';
import './PipelineHealth.css';

export default function PipelineHealth() {
  return (
    <div className="ui-card health-card">
      <div className="health-card__header">
        <h3 className="health-card__title">Pipeline Health</h3>
        <CheckCircle2 size={16} className="health-check-icon" />
      </div>

      <div className="health-card__list">
        {pipelineHealth.map((item, idx) => (
          <div key={idx} className="health-row">
            <span className="health-label">{item.label}</span>
            <div className="health-value-box">
              {item.status === 'online' && (
                <span className="health-dot health-dot--green"></span>
              )}
              <span className={`health-value ${item.status === 'synced' ? 'health-value--synced' : ''}`}>
                {item.value}
              </span>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
