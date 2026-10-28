import React from 'react';
import {
  Chart as ChartJS,
  ArcElement,
  Tooltip
} from 'chart.js';
import { Doughnut } from 'react-chartjs-2';
import { usePipeline } from '../data/PipelineContext';
import './DefectsByCategory.css';

ChartJS.register(ArcElement, Tooltip);

export default function DefectsByCategory() {
  const { defectsByCategory, systemMetrics } = usePipeline();
  const chartData = {
    labels: defectsByCategory.map((d) => d.name),
    datasets: [
      {
        data: defectsByCategory.map((d) => d.count),
        backgroundColor: defectsByCategory.map((d) => d.color),
        borderColor: '#ffffff',
        borderWidth: 2,
        hoverOffset: 4,
      },
    ],
  };

  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    cutout: '72%',
    plugins: {
      legend: {
        display: false,
      },
      tooltip: {
        backgroundColor: '#0f172a',
        padding: 10,
        callbacks: {
          label: (item) => {
            const count = item.raw;
            const pct = (systemMetrics.defectsDetected > 0 ? (count / systemMetrics.defectsDetected) * 100 : 0).toFixed(1);
            return ` ${item.label}: ${count} (${pct}%)`;
          }
        }
      }
    }
  };

  return (
    <div className="ui-card defects-cat-card">
      {/* Header */}
      <div className="defects-cat-card__header">
        <div className="defects-cat-card__title-row">
          <h2 className="defects-cat-card__title">Defects by Category</h2>
          <span className="defects-cat-card__badge">6 Standard Classes</span>
        </div>
        <p className="defects-cat-card__subtitle">
          Breakdown of {systemMetrics.defectsDetected} detected anomalies
        </p>
      </div>

      {/* Donut Chart with Centered Total */}
      <div className="defects-cat-card__chart-container">
        <div className="defects-cat-card__chart-wrapper">
          <Doughnut data={chartData} options={chartOptions} />
          <div className="defects-cat-card__center-text">
            <span className="center-num">{systemMetrics.defectsDetected}</span>
            <span className="center-label">TOTAL DEFECTS</span>
          </div>
        </div>
      </div>

      {/* Breakdown List with Progress Bars */}
      <div className="defects-cat-card__list">
        {defectsByCategory.map((item) => (
          <div key={item.key} className="defects-cat-item">
            <div className="defects-cat-item__top">
              <div className="defects-cat-item__left">
                <span className="cat-dot" style={{ backgroundColor: item.color }}></span>
                <span className="cat-name">{item.name}</span>
              </div>
              <span className="cat-stats">
                <strong>{item.count}</strong> ({item.percentage.toFixed(1)}%)
              </span>
            </div>
            <div className="defects-cat-item__bar-bg">
              <div
                className="defects-cat-item__bar-fill"
                style={{
                  width: `${item.percentage}%`, // Scaled for visual prominence
                  backgroundColor: item.barColor || item.color
                }}
              ></div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
