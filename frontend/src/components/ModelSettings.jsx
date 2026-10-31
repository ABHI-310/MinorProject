import React,{useState} from 'react';
import {api,usePipeline} from '../data/PipelineContext';
export default function ModelSettings(){
 const [reviewDemo,setReviewDemo]=useState(null);
 const {data}=usePipeline();const [timeline,setTimeline]=useState(null),[error,setError]=useState('');
 return <><h2>Models and Maintenance</h2><p>Checkpoint hashes are checked when models load. Scores use different evaluation datasets and label budgets.</p>
 <div className="pipeline-models">{data?.models?.map(m=><section key={m.model_type} className="ui-card pipeline-panel"><h3>{m.model_name}</h3><p>{m.available?'Available':'Missing'} · {m.method}</p><p>mAP50: {m.map_score!=null?(100*m.map_score).toFixed(2)+'%':'Unknown'} · mAP50–95: {m.map50_95!=null?(100*m.map50_95).toFixed(2)+'%':'Unknown'}</p><p>{m.evaluation}</p><p>Evaluated classes: {m.evaluated_class_ids.map(i=>m.classes[i]).join(', ')}</p><p>Inference image size: {m.image_size}</p>{m.operating_point&&<p>Validation at confidence {m.operating_point.confidence}: precision {(100*m.operating_point.precision).toFixed(2)}% · recall {(100*m.operating_point.recall).toFixed(2)}% · F1 {(100*m.operating_point.f1).toFixed(2)}%. NMS IoU: {m.nms_iou}.</p>}{m.limitations?.map(item=><p key={item}>{item}</p>)}<code className="pipeline-hash">{m.checkpoint_sha256}</code></section>)}</div>
 <section className="ui-card pipeline-panel"><h3>Inspection history · maintenance review</h3>
 <p>Uses the selected equipment's saved inspections. Report actual boards represented by each inspection; a single-board image represents one board.</p>
 <p>Status: {({insufficient_history:'Need six saved inspections',exposure_required:'Board counts required for all six inspections',settings_changed:'Model or confidence changed: collect six comparable inspections',review_required:'Review required: detection rate increased',no_rule_trigger:'No review rule triggered'})[data?.maintenance_trend?.status]||'Waiting for history'}</p>
 {data?.maintenance_trend?.recent_rate_per_100_boards!=null&&<p>Previous: {data.maintenance_trend.previous_rate_per_100_boards.toFixed(2)} · Recent: {data.maintenance_trend.recent_rate_per_100_boards.toFixed(2)} detections per 100 boards.</p>}
 <p>{data?.maintenance_trend?.rule}</p><p>A review alert can reflect output changes or detector errors; it does not diagnose a machine fault.</p>
 <h4>Synthetic review-alert example</h4><p>Show a fabricated increase without adding records to your inspection history.</p>
 <button className="topbar-btn topbar-btn--white" onClick={async()=>{try{setReviewDemo(await api('/maintenance/demo'));setError('');}catch(e){setError(e.message);}}}>Run synthetic review demo</button>
 {reviewDemo&&<div role="status"><p>{reviewDemo.description}</p><p>Synthetic result: {reviewDemo.trend.alert?'REVIEW REQUIRED':'No alert'} · Previous rate: {reviewDemo.trend.previous_rate_per_100_boards} · Recent rate: {reviewDemo.trend.recent_rate_per_100_boards} detections per 100 boards.</p><p>Demo observations, oldest first: {reviewDemo.inspections_oldest_first.map(row=>row.count).join(' → ')}. Nothing saved to inspection history.</p></div>}
 <h4>Simulated health outlook</h4><p>Status: {data?.rul?.status||'No history'}</p>
 {data?.rul?.health_score!=null&&<p>Simulated health: {(100*data.rul.health_score).toFixed(1)}% · Simulated RUL: {data.rul.rul_cycles} cycles ({data.rul.cycle_unit}).</p>}
 <p>{data?.rul?.note||'Estimates require known exposure and sufficient inspection history.'}</p>
 </section>
 <section className="ui-card pipeline-panel"><h3>Phase 3 · Synthetic maintenance demonstration</h3><p>This simulated lifecycle is separate from live inspection history. It is not a calibrated prediction for your equipment.</p>
 <button className="topbar-btn topbar-btn--white" onClick={async()=>{try{setTimeline(await api('/rul/synthetic/EQP_001'));setError('');}catch(e){setError(e.message);}}}>Load synthetic lifecycle</button>
 {error&&<p role="alert">{error}</p>}{timeline&&<><p>Synthetic equipment: {timeline.equipment_id} · Observed failure: {String(timeline.failure_observed)} · Failure cycle: {timeline.failure_cycle??'Not observed'} · {timeline.cycle_unit}</p><div className="pipeline-scroll"><table className="inspections-table"><thead><tr><th>Cycle</th><th>Health</th><th>RUL cycles</th></tr></thead><tbody>{timeline.timeline.filter((_,i)=>i%25===0).map((row,i)=><tr key={i}><td>{row.cycle}</td><td>{row.health_score?.toFixed(3)}</td><td>{row.rul_cycles??'Unknown'}</td></tr>)}</tbody></table></div></>}
 </section></>;
}
