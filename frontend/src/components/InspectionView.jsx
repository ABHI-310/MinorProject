import React, { useEffect, useRef, useState } from 'react';
import { api, usePipeline } from '../data/PipelineContext';
import './InspectionView.css';
export default function InspectionView({inspection,onInspected,onBackToOverview}) {
  const {equipment,model,confidence,refresh,inferenceBusy:busy,setInferenceBusy:setBusy}=usePipeline();
  const [file,setFile]=useState(null),[preview,setPreview]=useState('');
  const [error,setError]=useState(''),[exposure,setExposure]=useState(''),[mode,setMode]=useState('annotated');
  const [result,setResult]=useState(null);
  const event=useRef(null);
  useEffect(()=>{setResult(null);setError('');},[equipment,inspection?.id]);
  useEffect(()=>{event.current=null;},[file,equipment,model,confidence,exposure]);
  useEffect(()=>{
    if(!file){setPreview('');return;}
    const url=URL.createObjectURL(file);setPreview(url);return()=>URL.revokeObjectURL(url);
  },[file]);
  const current=result||inspection;
  async function submit(e) {
    e.preventDefault();setError('');
    if(!file){setError('Choose a PCB image first.');return;}
    if(file.size>10*1024*1024){setError('Choose an image smaller than 10 MB.');return;}
    if(!(confidence>=0.001&&confidence<=1)){setError('Confidence must be between 0.001 and 1.');return;}
    setBusy(true);
    try {
      event.current ||= crypto.randomUUID();
      const form=new FormData();form.append('file',file);form.append('equipment_id',equipment);
      form.append('model_type',model);form.append('confidence',confidence);form.append('event_id',event.current);
      form.append('annotate','true');if(exposure)form.append('boards_inspected',exposure);
      const value=await api('/detect',{method:'POST',body:form});
      setResult(value);event.current=null;await refresh();onInspected({id:value.inspection_id});
    } catch(e) {setError(e.message);} finally {setBusy(false);}
  }
  const original=current?.image_url;
  const annotated=current?.annotated_image_url||current?.annotated_image;
  return <div className="inspection-view">
    <div className="inspection-topbar"><div><button className="breadcrumb-link" onClick={onBackToOverview}>Overview</button><h2 className="inspection-title">PCB Inspection</h2><p>Run the verified {model} detector for {equipment}.</p></div></div>
    <form onSubmit={submit} className="ui-card pipeline-upload">
      <label>PCB image (JPEG or PNG)<input aria-label="PCB image" type="file" accept="image/jpeg,image/png" disabled={busy} onChange={e=>{setFile(e.target.files?.[0]||null);setResult(null);setError('');}}/></label>
      <label>Boards represented (optional)<input aria-label="Boards represented" type="number" min="1" max="1000000" step="1" value={exposure} onChange={e=>setExposure(e.target.value)} disabled={busy} placeholder="Only if known"/></label>
      <button type="submit" className="topbar-btn topbar-btn--brown" disabled={busy||!file}>{busy?'Running model…':'Run inspection'}</button>
      <p>Leave board exposure blank if unknown. No detections means the model found none at this threshold; it does not certify a defect-free board.</p>
      {error&&<p className="pipeline-error" role="alert">{error}</p>}
      {busy&&<p role="status">Running inference and saving the inspection…</p>}
    </form>
    {preview&&<details className="ui-card pipeline-panel"><summary>Selected image: {file.name}</summary><img className="pipeline-result-image" src={preview} alt="Selected PCB"/></details>}
    {!current?<div className="ui-card pipeline-panel">No inspections for this equipment yet. Upload an image to start.</div>:<>
      <div className="inspection-grid">
        <div className="inspection-grid__left"><div className="ui-card aoi-viewer-card">
          <div className="aoi-viewer-header"><h3 className="aoi-title">Inspection result</h3><div className="aoi-mode-toggles">
            <button className={`aoi-mode-btn ${mode==='annotated'?'aoi-mode-btn--active':''}`} onClick={()=>setMode('annotated')}>AI overlay</button>
            <button className={`aoi-mode-btn ${mode==='raw'?'aoi-mode-btn--active':''}`} onClick={()=>setMode('raw')}>Original</button>
          </div></div>
          {(mode==='annotated'?annotated||original:original)?<img className="pipeline-result-image" src={mode==='annotated'?annotated||original:original} alt={mode==='annotated'?'PCB detection overlay':'Original inspected PCB'}/>:<p className="pipeline-panel">Image unavailable for this older or edge inspection. New uploads keep their images.</p>}
          <div className="aoi-viewer-meta"><p>Inspection ID: {current.inspection_id||current.id}</p><p>{new Date(current.timestamp).toLocaleString()} · {current.model_type} · {current.count} detections · {current.inference_time_ms??'—'} ms</p><p>Threshold: {current.confidence_cutoff??'—'} · Resolution: {current.image_width??'—'} × {current.image_height??'—'}</p></div>
        </div></div>
        <div className="inspection-grid__right"><div className="ui-card pipeline-panel"><h3>Maintenance outlook</h3>
          <p>{current.rul?.status?.replaceAll('_',' ')||'Unavailable'}</p>
          <p>Health: {current.rul?.health_score!=null?current.rul.health_score.toFixed(3):'Unknown'}</p>
          <p>RUL: {current.rul?.rul_cycles!=null?`${current.rul.rul_cycles} cycles (simulated)`:'Unknown'}</p>
          <p>Requires at least 5 inspections and 500 known represented boards. Available estimates use simulator assumptions, not measured machine lifetime.</p>
          <button className="topbar-btn topbar-btn--white" onClick={()=>{
            const blob=new Blob([JSON.stringify(current,null,2)],{type:'application/json'});
            const url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download=`inspection-${current.inspection_id||current.id}.json`;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
          }}>Download result JSON</button>
        </div></div>
      </div>
      <div className="ui-card pipeline-panel"><h3>Detected defects</h3><div className="inspections-table-wrap"><table className="inspections-table"><thead><tr><th>Defect</th><th>Confidence</th><th>Bounding box (pixels: x1, y1, x2, y2)</th></tr></thead><tbody>
        {current.detections.map((d,i)=><tr key={i}><td>{d.class_name.replaceAll('_',' ')}</td><td>{(d.confidence*100).toFixed(1)}%</td><td>{d.bbox.map(v=>v.toFixed(1)).join(', ')}</td></tr>)}
        {!current.count&&<tr><td colSpan="3">No detections at this confidence threshold.</td></tr>}
      </tbody></table></div></div>
    </>}
  </div>;
}
