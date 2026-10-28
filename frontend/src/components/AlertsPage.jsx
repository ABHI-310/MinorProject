import React from 'react';
import {usePipeline} from '../data/PipelineContext';
import RecentAlerts from './RecentAlerts';
export default function AlertsPage({onNavigateToInspection}) {
 const {systemMetrics}=usePipeline();
 return <><h2>Critical Defect Review</h2><p>{systemMetrics.criticalAlerts} short/open-circuit detections in the last 30 days. Recent records below are limited to the latest 100 inspections.</p><RecentAlerts onReviewAnomaly={onNavigateToInspection}/></>;
}
