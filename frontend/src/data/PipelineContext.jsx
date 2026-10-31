import React, { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState } from 'react';

const PipelineContext = createContext(null);
export const categories = [
  ['missing_hole','Missing Hole','#3b82f6'], ['mouse_bite','Mouse Bite','#ec4899'],
  ['open_circuit','Open Circuit','#ef4444'], ['short','Short','#ea580c'],
  ['spur','Spur','#8b5cf6'], ['spurious_copper','Spurious Copper','#14b8a6']
];
export async function api(path, options={}) {
  const response=await fetch(`/api${path}`, options);
  if (!response.ok) {
    let detail;
    try { detail=(await response.json()).detail; } catch { detail=response.statusText; }
    throw new Error(typeof detail==='string' ? detail : JSON.stringify(detail));
  }
  return response.json();
}
export function PipelineProvider({children}) {
  const [equipment,setEquipment]=useState('EQP_001');
  const [model,setModel]=useState('source');
  const [confidence,setConfidence]=useState(0.25);
  const [inferenceBusy,setInferenceBusy]=useState(false);
  const [data,setData]=useState(null);
  const [error,setError]=useState('');
  const [lastSync,setLastSync]=useState(null);
  const request=useRef(null);
  const currentEquipment=useRef(equipment);currentEquipment.current=equipment;
  const refresh=useCallback(async()=>{
    if(currentEquipment.current!==equipment)return;
    request.current?.abort();
    const controller=new AbortController(); request.current=controller;
    try {
      const value=await api(`/dashboard?equipment_id=${encodeURIComponent(equipment)}&days=30`,{signal:controller.signal});
      if (controller.signal.aborted || currentEquipment.current!==equipment) return;
      setData(value);setError('');setLastSync(new Date());
    } catch(e) { if(e.name!=='AbortError') setError(e.message); }
  },[equipment]);
  useEffect(()=>{
    setData(null);setLastSync(null);refresh();
    const timer=setInterval(refresh,30000);
    return ()=>{clearInterval(timer);request.current?.abort();};
  },[refresh]);
  const value=useMemo(()=>{
    const counts=data?.defect_counts||{};
    const total=data?.detection_count||0;
    const recentInspections=(data?.history||[]).map(item=>{
      const best=[...item.detections].sort((a,b)=>b.confidence-a.confidence)[0];
      const category=categories.find(c=>c[0]===best?.class_name);
      return {...item,id:item.inspection_id,defectKey:best?.class_name||'passed',
        defectType:best ? (category?.[1]||best.class_name)+(item.count>1?` + ${item.count-1}`:'') : 'No detection',
        confidence:best ? (best.confidence*100).toFixed(1) : '—',color:category?.[2]||'#10b981',badgeBg:'#fff7ed',
        status:item.count?'Defect Found':'No detection',statusType:item.count?'defect':'passed',
        thumbnail:item.annotated_image_url||item.image_url||'',
        timestampLabel:new Date(item.timestamp).toLocaleString(),timeAgo:item.model_type,
        details:`${item.count} detections · ${item.model_type} · ${item.equipment_id}`};
    });
    const selected=data?.models?.find(m=>m.model_type===model);
    const last=recentInspections[0];
    const inspected=data?.inspection_count||0;
    const critical=data?.critical_detection_count||0;
    const systemMetrics={
      inspected,inspectedFormatted:data?inspected.toLocaleString():'—',inspectedTrend:'Last 30 days',
      inspectedSubtitle:`Saved inspections · ${equipment}`,defectsDetected:data?total:'—',
      defectsRate:'Actual detections',defectsSubtitle:'Detected boxes in the selected equipment window',
      defectiveBoards:data?.defective_inspections??'—',
      defectsPerBoard:inspected?`${(total/inspected).toFixed(2)} detections/inspection`:'No inspections yet',
      defectYield:inspected?`${(100*(data.defective_inspections/inspected)).toFixed(1)}% with detections`:'No data',
      criticalAlerts:critical,criticalFormatted:data?String(critical):'—',
      criticalSubtitle:'Short and open-circuit detections',
      inferenceTime:last?.inference_time_ms!=null?`${last.inference_time_ms.toFixed(0)} ms`:'Not run',
      systemStatus:error?'UNAVAILABLE':data?'ONLINE':'CONNECTING',
      modelName:selected?.model_name||model,
      f1Score:selected?.map_score!=null?`${(100*selected.map_score).toFixed(2)}% dev/holdout mAP50`:'Evaluation unavailable',
      confidenceThreshold:`${(confidence*100).toFixed(0)}% (new uploads)`,inferenceWindow:'30 days',
      activeLine:equipment,dateRange:'Last 30 days · UTC aggregation'
    };
    const defectsByCategory=categories.map(([key,name,color])=>({key,name,color,count:counts[key]||0,
      percentage:total?100*(counts[key]||0)/total:0}));
    const trendData={};
    for(const days of [7,14,30]) {
      const keys=Array.from({length:days},(_,i)=>{
        const date=new Date();date.setUTCDate(date.getUTCDate()-(days-1-i));return date.toISOString().slice(0,10);
      });
      trendData[`${days}D`]={labels:keys,datasets:categories.map(([key,label,borderColor])=>{
        const series=keys.map(day=>data?.daily_defects?.[day]?.[key]||0);
        return {label,borderColor,data:series,total:series.reduce((a,b)=>a+b,0),tension:0.3,pointRadius:2};
      })};
    }
    const recentAlerts=recentInspections.filter(i=>i.detections.some(d=>['short','open_circuit'].includes(d.class_name)))
      .map(i=>({id:i.id,type:'CRITICAL DEFECT DETECTED',severity:'critical',timeAgo:i.timestampLabel,
        description:i.details,boardId:i.id,hasAction:true,actionText:'Review Inspection',inspectionRef:i.id}));
    const pipelineHealth=[
      {label:'API Core',value:error?'Unavailable':data?'Connected':'Connecting',status:error?'error':'online'},
      {label:'Database',value:data?.health?.database||'Unknown',status:'normal'},
      {label:'AI Detector',value:`${data?.health?.available_models?.length||0} checkpoints · ${data?.health?.device||'unknown device'}`,status:'normal'},
      {label:'Last Sync',value:lastSync?.toLocaleTimeString()||'Waiting',status:'synced'}
    ];
    return {data,error,lastSync,equipment,setEquipment,model,setModel,confidence,setConfidence,refresh,
      inferenceBusy,setInferenceBusy,
      systemMetrics,recentInspections,defectsByCategory,trendData,recentAlerts,pipelineHealth};
  },[data,error,lastSync,equipment,model,confidence,refresh,inferenceBusy]);
  return <PipelineContext.Provider value={value}>{children}</PipelineContext.Provider>;
}
export function usePipeline(){return useContext(PipelineContext);}
