import React, { useState } from 'react';
import { usePipeline } from '../data/PipelineContext';
export default function PipelineControls({onInspect}) {
  const {equipment,setEquipment,data,error,refresh,model,setModel,confidence,setConfidence,inferenceBusy}=usePipeline();
  const [draft,setDraft]=useState(equipment);
  return <section className="ui-card pipeline-controls">
    <form onSubmit={e=>{e.preventDefault();setEquipment(draft);}}>
      <label>Equipment ID<input aria-label="Equipment ID" disabled={inferenceBusy} value={draft} onChange={e=>setDraft(e.target.value)} pattern="[A-Za-z0-9][A-Za-z0-9_.\-]{0,49}" required maxLength={50}/></label>
      <button className="topbar-btn topbar-btn--white" type="submit" disabled={inferenceBusy}>Apply equipment</button>
    </form>
    <label>Detector<select aria-label="Detector" disabled={inferenceBusy} value={model} onChange={e=>setModel(e.target.value)}>
      <option value="source">Full bare board · all 6 labels</option><option value="target">DsPCBSD+ close-up · 5 labels</option>
    </select></label>
    <label>Confidence threshold<input aria-label="Confidence threshold" disabled={inferenceBusy} type="number" min="0.001" max="1" step="0.001" value={confidence} onChange={e=>setConfidence(Number(e.target.value))}/></label>
    <button className="topbar-btn topbar-btn--brown" onClick={onInspect}>Inspect a PCB</button>
    <p>{model==='source' ? 'All six labels are enabled. Use this checkpoint for bare full-board images resembling PKU training data.' : 'Close-up checkpoint: missing_hole is not supported. Do not use this model for full-board photographs.'} A copper-trace break is open_circuit; general board or component cracks are outside these labels.</p>
    {!data && !error && <p role="status">Connecting to API…</p>}
    {error && <p role="alert" className="pipeline-error">API unavailable: {error}. Displayed records may be stale. <button onClick={refresh}>Retry</button></p>}
  </section>;
}
