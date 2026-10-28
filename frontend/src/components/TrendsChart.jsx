import React, { useState } from 'react';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Tooltip,
  Legend,
  Filler,
} from 'chart.js';
import { Line } from 'react-chartjs-2';
import { usePipeline } from '../data/PipelineContext';
import './TrendsChart.css';

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Tooltip,
  Legend,
  Filler
);

export default function TrendsChart() {
  const { trendData, systemMetrics } = usePipeline();
  const [range, setRange] = useState('7D');
  const currentTrend = trendData[range] || trendData['7D'];

  const chartData = {
    labels: currentTrend.labels,
    datasets: currentTrend.datasets,
  };

  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        display: false, // We render the custom legend exactly as shown in mockup
      },
      tooltip: {
        backgroundColor: '#0f172a',
        titleColor: '#38bdf8',
        bodyColor: '#f8fafc',
        borderColor: '#334155',
        borderWidth: 1,
        padding: 12,
        boxPadding: 6,
        usePointStyle: true,
        callbacks: {
          title: (items) => {
            const label = items[0]?.label || '';
            return `${label} UTC`;
          },
          label: (item) => {
            return ` ${item.dataset.label}: ${item.raw}`;
          }
        }
      },
    },
    scales: {
      x: {
        grid: {
          color: '#f1f5f9',
          tickColor: 'transparent',
        },
        ticks: {
          color: '#64748b',
          font: { size: 10, family: 'Inter' },
        },
      },
      y: {
        min: 0,
        
        grid: {
          color: '#f1f5f9',
        },
        ticks: {
          color: '#94a3b8',
          font: { size: 10, family: 'Inter' },
          stepSize: 10,
        },
        border: {
          dash: [4, 4],
        }
      },
    },
    interaction: {
      mode: 'index',
      intersect: false,
    },
  };

  return (
    <div className="ui-card trends-card">
      {/* Top Header */}
      <div className="trends-card__header">
        <div className="trends-card__title-group">
          <div className="trends-card__title-row">
            <h2 className="trends-card__title">Defect Detection Trends</h2>
            <span className="trends-card__badge">{range} Analysis</span>
          </div>
          <p className="trends-card__subtitle">
            Daily defect frequency by class over the last {range === '7D' ? '7 days' : range === '14D' ? '14 days' : '30 days'}
          </p>
        </div>

        {/* Segmented Control */}
        <div className="trends-card__controls">
          {['7D', '14D', '30D'].map((r) => (
            <button
              key={r}
              className={`trends-card__btn ${range === r ? 'trends-card__btn--active' : ''}`}
              onClick={() => setRange(r)}
            >
              {r}
            </button>
          ))}
        </div>
      </div>

      {/* Legend Row */}
      <div className="trends-card__legend">
        {currentTrend.datasets.map((ds) => (
          <div key={ds.label} className="trends-card__legend-item">
            <span className="legend-dash" style={{ backgroundColor: ds.borderColor }}></span>
            <span className="legend-label">{ds.label}</span>
            <span className="legend-count">({ds.total})</span>
          </div>
        ))}
      </div>

      {/* Main Chart Area */}
      <div className="trends-card__chart-wrap">
        <Line data={chartData} options={chartOptions} />
      </div>

      {/* Footer Annotations */}
      <div className="trends-card__footer">
        <span className="trends-card__footer-left">
          Inference window: {systemMetrics.inferenceWindow}
        </span>
        <span className="trends-card__footer-right">
          Confidence threshold: {systemMetrics.confidenceThreshold}
        </span>
      </div>
    </div>
  );
}
