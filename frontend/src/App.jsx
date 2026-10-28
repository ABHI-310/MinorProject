import React, { useState } from 'react';
import Sidebar from './components/Sidebar';
import Header from './components/Header';
import RuntimeBanner from './components/RuntimeBanner';
import MetricsPanel from './components/MetricsPanel';
import TrendsChart from './components/TrendsChart';
import DefectsByCategory from './components/DefectsByCategory';
import RecentInspections from './components/RecentInspections';
import RecentAlerts from './components/RecentAlerts';
import PipelineHealth from './components/PipelineHealth';
import InspectionView from './components/InspectionView';
import DefectAnalytics from './components/DefectAnalytics';
import AlertsPage from './components/AlertsPage';
import { PipelineProvider, usePipeline } from './data/PipelineContext';
import PipelineControls from './components/PipelineControls';
import ModelSettings from './components/ModelSettings';
import './Pipeline.css';
import './App.css';

export default function App() { return <PipelineProvider><Dashboard /></PipelineProvider>; }

function Dashboard() {
  const { recentInspections, refresh } = usePipeline();
  const [activeNav, setActiveNav] = useState('overview'); // 'overview' | 'inspections' | 'analytics' | 'alerts'
  const [currentInspection, setCurrentInspection] = useState(null);
  const [isSidebarCollapsed, setIsSidebarCollapsed] = useState(false);

  const handleOpenInspection = (inspection) => {
    setCurrentInspection(inspection || recentInspections[0]);
    setActiveNav('inspections');
  };

  const handleReviewAnomaly = (inspectionId) => {
    const found = recentInspections.find((i) => i.id === inspectionId);
    handleOpenInspection(found || recentInspections[0]);
  };

  const handleActionReq = () => {
    setActiveNav('alerts');
  };

  return (
    <div className="app-container">
      {/* 1. Left Sidebar */}
      <Sidebar
        activeNav={activeNav}
        onSelectNav={setActiveNav}
        isCollapsed={isSidebarCollapsed}
        onToggleCollapse={() => setIsSidebarCollapsed((prev) => !prev)}
      />

      {/* 2. Main Content Area */}
      <div className="main-wrapper">
        <Header onSync={refresh} />

        <main className="dashboard-content">
          <PipelineControls onInspect={() => setActiveNav('inspections')} />
          {activeNav === 'settings' ? <ModelSettings /> : activeNav === 'inspections' ? (
            <InspectionView
              inspection={recentInspections.find(i=>i.id===currentInspection?.id) || recentInspections[0]}
              onInspected={handleOpenInspection}
              onBackToOverview={() => setActiveNav('overview')}
            />
          ) : activeNav === 'analytics' ? (
            <DefectAnalytics
              onBackToOverview={() => setActiveNav('overview')}
              onSelectDefectType={(type) => {
                const match = recentInspections.find((i) => i.defectKey === type);
                handleOpenInspection(match || recentInspections[0]);
              }}
            />
          ) : activeNav === 'alerts' ? (
            <AlertsPage
              onBackToOverview={() => setActiveNav('overview')}
              onNavigateToInspection={(boardId) => {
                const match = recentInspections.find((i) => i.id === boardId);
                handleOpenInspection(match || recentInspections[0]);
              }}
            />
          ) : (
            <>
              {/* Runtime Profile Banner */}
              <RuntimeBanner />

              {/* 4 Top KPI Cards */}
              <MetricsPanel onActionReq={handleActionReq} />

              {/* Middle Row: Trends (2/3) + Defects by Category (1/3) */}
              <div className="dashboard-grid-row">
                <div className="dashboard-col-trends">
                  <TrendsChart />
                </div>
                <div className="dashboard-col-category">
                  <DefectsByCategory />
                </div>
              </div>

              {/* Bottom Row: Recent Inspections (2/3) + Alerts & Pipeline (1/3) */}
              <div className="dashboard-grid-row">
                <div className="dashboard-col-inspections">
                  <RecentInspections onSelectInspection={handleOpenInspection} />
                </div>
                <div className="dashboard-col-sidebar-stack">
                  <RecentAlerts onReviewAnomaly={handleReviewAnomaly} />
                  <PipelineHealth />
                </div>
              </div>
            </>
          )}
        </main>
      </div>
    </div>
  );
}
