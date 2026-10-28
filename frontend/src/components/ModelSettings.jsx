import React,{useState} from 'react';
import {api,usePipeline} from '../data/PipelineContext';
export default function ModelSettings(){
 const {data}=usePipeline();const [timeline,setTimeline]=useState(null),[error,setError]=useState('');
 return <><h2>Models and Maintenance</h2><p>Both checkpoints are verified by SHA-256. Source and target scores use different datasets.</p>
 <div className="pipeline-models">{data?.models?.map(m=><section key={m.model_type} className="ui-card pipeline-panel"><h3>{m.model_name}</h3><p>{m.available?'Available':'Missing'} · {m.method}</p><p>mAP50: {m.map_score!=null?(100*m.map_score).toFixed(2)+'%':'Unknown'} · mAP50–95: {m.map50_95!=null?(100*m.map50_95).toFixed(2)+'%':'Unknown'}</p><p>{m.evaluation}</p><p>Evaluated classes: {m.evaluated_class_ids.map(i=>m.classes[i]).join(', ')}</p><p>Inference image size: {m.image_size}</p><code className="pipeline-hash">{m.checkpoint_sha256}</code></section>)}</div>
 <section className="ui-card pipeline-panel"><h3>Phase 3 · Synthetic maintenance demonstration</h3><p>This simulated lifecycle is separate from live inspection history. It is not a calibrated prediction for your equipment.</p>
 <button className="topbar-btn topbar-btn--white" onClick={async()=>{try{setTimeline(await api('/rul/synthetic/EQP_001'));setError('');}catch(e){setError(e.message);}}}>Load synthetic lifecycle</button>
 {error&&<p role="alert">{error}</p>}{timeline&&<><p>Synthetic equipment: {timeline.equipment_id} · Observed failure: {String(timeline.failure_observed)} · Failure cycle: {timeline.failure_cycle??'Not observed'} · {timeline.cycle_unit}</p><div className="pipeline-scroll"><table className="inspections-table"><thead><tr><th>Cycle</th><th>Health</th><th>RUL cycles</th></tr></thead><tbody>{timeline.timeline.filter((_,i)=>i%25===0).map((row,i)=><tr key={i}><td>{row.cycle}</td><td>{row.health_score?.toFixed(3)}</td><td>{row.rul_cycles??'Unknown'}</td></tr>)}</tbody></table></div></>}
 </section></>;
}
