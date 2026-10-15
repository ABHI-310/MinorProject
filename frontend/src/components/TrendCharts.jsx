import { useMemo } from 'react';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  ArcElement,
  Tooltip,
  Legend,
  Filler,
} from 'chart.js';
import { Line, Doughnut, Bar } from 'react-chartjs-2';
import { TrendingUp, PieChart, Activity, BarChart3 } from 'lucide-react';
import { trainingHistory } from '../data/trainingHistory';
import { classDistribution } from '../data/datasetStats';
import { perClassMetrics } from '../data/modelMetrics';
import { hourlyDefectTrend } from '../data/mockDetections';
import './TrendCharts.css';

// Register Chart.js components
ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  ArcElement,
  Tooltip,
  Legend,
  Filler
);

// Global Chart.js defaults for dark theme
ChartJS.defaults.color = '#94a3b8';
ChartJS.defaults.borderColor = 'rgba(148, 163, 184, 0.08)';
ChartJS.defaults.font.family = "'Inter', sans-serif";
ChartJS.defaults.font.size = 11;

const commonLineOptions = {
  responsive: true,
  maintainAspectRatio: false,
  interaction: { mode: 'index', intersect: false },
  plugins: {
    legend: {
      position: 'top',
      labels: { boxWidth: 8, usePointStyle: true, pointStyle: 'circle', padding: 16 },
    },
    tooltip: {
      backgroundColor: 'rgba(15, 23, 42, 0.9)',
      borderColor: 'rgba(99, 115, 171, 0.2)',
      borderWidth: 1,
      titleFont: { weight: '600' },
      padding: 10,
      cornerRadius: 8,
    },
  },
  scales: {
    x: { grid: { display: false }, ticks: { maxTicksLimit: 10 } },
    y: { grid: { color: 'rgba(148, 163, 184, 0.05)' }, beginAtZero: true },
  },
};

export default function TrendCharts() {
  // 1. Defect Count Trend (line chart)
  const defectTrendData = useMemo(() => ({
    labels: hourlyDefectTrend.map((h) => h.hour),
    datasets: [
      {
        label: 'Total',
        data: hourlyDefectTrend.map((h) => h.total),
        borderColor: '#3b82f6',
        backgroundColor: 'rgba(59, 130, 246, 0.1)',
        fill: true,
        tension: 0.4,
        pointRadius: 2,
        pointHoverRadius: 5,
        borderWidth: 2,
      },
      {
        label: 'Critical',
        data: hourlyDefectTrend.map((h) => h.critical),
        borderColor: '#ef4444',
        backgroundColor: 'rgba(239, 68, 68, 0.08)',
        fill: true,
        tension: 0.4,
        pointRadius: 2,
        pointHoverRadius: 5,
        borderWidth: 2,
      },
    ],
  }), []);

  // 2. Class Distribution (doughnut)
  const classDistData = useMemo(() => ({
    labels: classDistribution.map((c) => c.name.replace(/_/g, ' ')),
    datasets: [{
      data: classDistribution.map((c) => c.instances),
      backgroundColor: classDistribution.map((c) => c.color),
      borderColor: 'rgba(15, 23, 42, 0.8)',
      borderWidth: 2,
      hoverOffset: 8,
    }],
  }), []);

  const doughnutOptions = {
    responsive: true,
    maintainAspectRatio: false,
    cutout: '65%',
    plugins: {
      legend: {
        position: 'right',
        labels: { boxWidth: 10, usePointStyle: true, pointStyle: 'circle', padding: 12, font: { size: 10 } },
      },
      tooltip: {
        backgroundColor: 'rgba(15, 23, 42, 0.9)',
        borderColor: 'rgba(99, 115, 171, 0.2)',
        borderWidth: 1,
        padding: 10,
        cornerRadius: 8,
      },
    },
  };

  // 3. Training Loss Curve (line)
  const trainingLossData = useMemo(() => ({
    labels: trainingHistory.map((h) => `Ep ${h.epoch}`),
    datasets: [
      {
        label: 'Train Box Loss',
        data: trainingHistory.map((h) => h.trainBoxLoss),
        borderColor: '#3b82f6',
        tension: 0.4,
        pointRadius: 3,
        pointHoverRadius: 6,
        borderWidth: 2,
      },
      {
        label: 'Val Box Loss',
        data: trainingHistory.map((h) => h.valBoxLoss),
        borderColor: '#8b5cf6',
        borderDash: [5, 3],
        tension: 0.4,
        pointRadius: 3,
        pointHoverRadius: 6,
        borderWidth: 2,
      },
      {
        label: 'Train Cls Loss',
        data: trainingHistory.map((h) => h.trainClsLoss),
        borderColor: '#10b981',
        tension: 0.4,
        pointRadius: 3,
        pointHoverRadius: 6,
        borderWidth: 2,
      },
      {
        label: 'Val Cls Loss',
        data: trainingHistory.map((h) => h.valClsLoss),
        borderColor: '#f59e0b',
        borderDash: [5, 3],
        tension: 0.4,
        pointRadius: 3,
        pointHoverRadius: 6,
        borderWidth: 2,
      },
    ],
  }), []);

  // 4. Per-class F1 Score (bar)
  const f1BarData = useMemo(() => ({
    labels: perClassMetrics.map((c) => c.name.replace(/_/g, ' ')),
    datasets: [{
      label: 'F1 Score',
      data: perClassMetrics.map((c) => c.f1),
      backgroundColor: [
        'rgba(59, 130, 246, 0.7)',
        'rgba(139, 92, 246, 0.7)',
        'rgba(239, 68, 68, 0.7)',
        'rgba(245, 158, 11, 0.7)',
        'rgba(16, 185, 129, 0.7)',
        'rgba(236, 72, 153, 0.7)',
      ],
      borderColor: [
        '#3b82f6',
        '#8b5cf6',
        '#ef4444',
        '#f59e0b',
        '#10b981',
        '#ec4899',
      ],
      borderWidth: 1,
      borderRadius: 6,
      borderSkipped: false,
    }],
  }), []);

  const barOptions = {
    responsive: true,
    maintainAspectRatio: false,
    indexAxis: 'y',
    plugins: {
      legend: { display: false },
      tooltip: {
        backgroundColor: 'rgba(15, 23, 42, 0.9)',
        borderColor: 'rgba(99, 115, 171, 0.2)',
        borderWidth: 1,
        padding: 10,
        cornerRadius: 8,
        callbacks: {
          label: (ctx) => `F1: ${ctx.parsed.x.toFixed(4)}`,
        },
      },
    },
    scales: {
      x: { grid: { color: 'rgba(148, 163, 184, 0.05)' }, beginAtZero: true, max: 0.06 },
      y: { grid: { display: false }, ticks: { font: { size: 10 } } },
    },
  };

  return (
    <div className="trend-charts__grid">
      <div className="chart-card">
        <div className="chart-card__header">
          <div className="chart-card__title">
            <TrendingUp /> Defect Count Trend (24h)
          </div>
        </div>
        <div className="chart-card__body">
          <Line data={defectTrendData} options={commonLineOptions} />
        </div>
      </div>

      <div className="chart-card">
        <div className="chart-card__header">
          <div className="chart-card__title">
            <PieChart /> Class Distribution
          </div>
        </div>
        <div className="chart-card__body">
          <Doughnut data={classDistData} options={doughnutOptions} />
        </div>
      </div>

      <div className="chart-card">
        <div className="chart-card__header">
          <div className="chart-card__title">
            <Activity /> Training Loss Curves
          </div>
        </div>
        <div className="chart-card__body">
          <Line data={trainingLossData} options={commonLineOptions} />
        </div>
      </div>

      <div className="chart-card">
        <div className="chart-card__header">
          <div className="chart-card__title">
            <BarChart3 /> Per-Class F1 Score
          </div>
        </div>
        <div className="chart-card__body">
          <Bar data={f1BarData} options={barOptions} />
        </div>
      </div>
    </div>
  );
}
