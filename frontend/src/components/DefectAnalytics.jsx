import React from 'react';
import {usePipeline} from '../data/PipelineContext';
import TrendsChart from './TrendsChart';
import DefectsByCategory from './DefectsByCategory';
export default function DefectAnalytics({onSelectDefectType}) {
 const {defectsByCategory,recentInspections}=usePipeline();
 return <><h2>Defect Analytics</h2><p>Actual saved detections for the selected equipment. Counts cover the last 30 days.</p>
 <div className="dashboard-grid-row"><TrendsChart/><DefectsByCategory/></div>
 <div className="ui-card pipeline-panel"><h3>Review a defect category</h3><p>Open a matching record from the latest 100 inspections.</p><div className="pipeline-actions">{defectsByCategory.map(d=><button disabled={!recentInspections.some(i=>i.defectKey===d.key)} className="topbar-btn topbar-btn--white" key={d.key} onClick={()=>onSelectDefectType(d.key)}>{d.name}: {d.count}</button>)}</div></div></>;
}
