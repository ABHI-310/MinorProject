/**
 * PCB Sentinel / Smart Maintenance Mock Data
 * Matching the production overview design language and metrics.
 */

export const systemMetrics = {
  inspected: 1284,
  inspectedFormatted: "1,284",
  inspectedTrend: "+12.4%",
  inspectedSubtitle: "Across 3 SMT assembly lines",
  
  defectsDetected: 186,
  defectsRate: "-4.1% rate",
  defectsSubtitle: "Total anomaly instances detected",
  
  defectiveBoards: 142,
  defectsPerBoard: "1.31 defects/board",
  defectYield: "11.05% Defect Yield",
  
  criticalAlerts: 8,
  criticalFormatted: "08",
  criticalSubtitle: "5 Shorts, 3 Opens pending review",
  
  inferenceTime: "24ms",
  systemStatus: "ONLINE",
  modelName: "YOLOv8-PCB-v2",
  f1Score: "99.4% F1",
  confidenceThreshold: ">88.5%",
  inferenceWindow: "168 hours",
  activeLine: "SMT LINE 04",
  dateRange: "Last 7 Days (May 14 - May 20, 2025)"
};

export const defectsByCategory = [
  { name: "Short", key: "short", count: 54, percentage: 29.0, color: "#ea580c", barColor: "#f97316" },
  { name: "Missing Hole", key: "missing_hole", count: 38, percentage: 20.4, color: "#3b82f6", barColor: "#60a5fa" },
  { name: "Open", key: "open_circuit", count: 32, percentage: 17.2, color: "#ef4444", barColor: "#f87171" },
  { name: "Mouse Bite", key: "mouse_bite", count: 26, percentage: 14.0, color: "#ec4899", barColor: "#f472b6" },
  { name: "Spur", key: "spur", count: 20, percentage: 10.8, color: "#8b5cf6", barColor: "#a78bfa" },
  { name: "Spurious Copper", key: "spurious_copper", count: 16, percentage: 8.6, color: "#14b8a6", barColor: "#2dd4bf" },
];

export const trendData = {
  "7D": {
    labels: ["May 14", "May 15", "May 16", "May 17", "May 18", "May 19", "May 20"],
    datasets: [
      {
        label: "Open",
        total: 32,
        data: [5, 7, 8, 9, 8, 7, 5],
        borderColor: "#ef4444",
        backgroundColor: "rgba(239, 68, 68, 0.08)",
        fill: true,
        tension: 0.35,
        borderWidth: 2,
        pointRadius: [2, 2, 2, 2, 2, 4, 2],
        pointHoverRadius: 6,
      },
      {
        label: "Short",
        total: 54,
        data: [7, 10, 16, 14, 18, 18, 15],
        borderColor: "#ea580c",
        backgroundColor: "rgba(234, 88, 12, 0.12)",
        fill: true,
        tension: 0.35,
        borderWidth: 2.5,
        pointRadius: [2, 2, 2, 2, 2, 4, 2],
        pointHoverRadius: 6,
      },
      {
        label: "Mouse Bite",
        total: 26,
        data: [6, 8, 9, 7, 6, 4, 4],
        borderColor: "#ec4899",
        backgroundColor: "rgba(236, 72, 153, 0.06)",
        fill: true,
        tension: 0.35,
        borderWidth: 2,
        pointRadius: [2, 2, 2, 2, 2, 4, 2],
        pointHoverRadius: 6,
      },
      {
        label: "Missing Hole",
        total: 38,
        data: [6, 9, 11, 15, 12, 11, 13],
        borderColor: "#3b82f6",
        backgroundColor: "rgba(59, 130, 246, 0.08)",
        fill: true,
        tension: 0.35,
        borderWidth: 2,
        pointRadius: [2, 2, 2, 2, 2, 4, 2],
        pointHoverRadius: 6,
      },
    ]
  },
  "14D": {
    labels: ["May 7", "May 9", "May 11", "May 13", "May 15", "May 17", "May 19", "May 20"],
    datasets: [
      {
        label: "Open",
        total: 62,
        data: [8, 9, 6, 7, 8, 9, 7, 8],
        borderColor: "#ef4444",
        backgroundColor: "rgba(239, 68, 68, 0.08)",
        fill: true,
        tension: 0.35,
        borderWidth: 2,
      },
      {
        label: "Short",
        total: 98,
        data: [12, 15, 11, 10, 16, 14, 18, 15],
        borderColor: "#ea580c",
        backgroundColor: "rgba(234, 88, 12, 0.12)",
        fill: true,
        tension: 0.35,
        borderWidth: 2.5,
      },
      {
        label: "Mouse Bite",
        total: 51,
        data: [7, 8, 6, 8, 9, 7, 4, 5],
        borderColor: "#ec4899",
        backgroundColor: "rgba(236, 72, 153, 0.06)",
        fill: true,
        tension: 0.35,
        borderWidth: 2,
      },
      {
        label: "Missing Hole",
        total: 74,
        data: [9, 10, 8, 9, 11, 15, 11, 13],
        borderColor: "#3b82f6",
        backgroundColor: "rgba(59, 130, 246, 0.08)",
        fill: true,
        tension: 0.35,
        borderWidth: 2,
      },
    ]
  },
  "30D": {
    labels: ["Apr 20", "Apr 25", "Apr 30", "May 5", "May 10", "May 15", "May 20"],
    datasets: [
      {
        label: "Open",
        total: 135,
        data: [18, 22, 19, 21, 24, 25, 20],
        borderColor: "#ef4444",
        backgroundColor: "rgba(239, 68, 68, 0.08)",
        fill: true,
        tension: 0.35,
        borderWidth: 2,
      },
      {
        label: "Short",
        total: 210,
        data: [26, 31, 28, 34, 39, 44, 38],
        borderColor: "#ea580c",
        backgroundColor: "rgba(234, 88, 12, 0.12)",
        fill: true,
        tension: 0.35,
        borderWidth: 2.5,
      },
      {
        label: "Mouse Bite",
        total: 112,
        data: [15, 18, 16, 17, 19, 18, 14],
        borderColor: "#ec4899",
        backgroundColor: "rgba(236, 72, 153, 0.06)",
        fill: true,
        tension: 0.35,
        borderWidth: 2,
      },
      {
        label: "Missing Hole",
        total: 160,
        data: [20, 23, 21, 25, 29, 31, 27],
        borderColor: "#3b82f6",
        backgroundColor: "rgba(59, 130, 246, 0.08)",
        fill: true,
        tension: 0.35,
        borderWidth: 2,
      },
    ]
  }
};

export const recentInspections = [
  {
    id: "#PCB-2025-0842",
    defectType: "Short",
    defectKey: "short",
    color: "#f59e0b",
    badgeBg: "#fef3c7",
    confidence: 98.2,
    timestamp: "14:28:12",
    timeAgo: "4m ago",
    status: "Defect Found",
    statusType: "defect",
    thumbnail: "/prediction_samples/pred_01_short_01_jpg.rf.543b3693d0c3f26d871a940538ed0ba8.jpg",
    details: "Trace bridge on Layer 2 SMT pad - bridging adjacent signal lanes",
    layer: "Layer 2 Signal",
    lot: "Lot B4-8842",
    operator: "Auto-AOI-01"
  },
  {
    id: "#PCB-2025-0841",
    defectType: "Mouse Bite",
    defectKey: "mouse_bite",
    color: "#ec4899",
    badgeBg: "#fce7f3",
    confidence: 96.4,
    timestamp: "14:24:05",
    timeAgo: "8m ago",
    status: "Review Req.",
    statusType: "review",
    thumbnail: "/prediction_samples/pred_01_mouse_bite_05_jpg.rf.f4cded1996897893b5e0216f9a365cb8.jpg",
    details: "Irregular copper boundary erosion exceeding 0.12mm tolerance",
    layer: "Top Silkscreen/Trace",
    lot: "Lot B4-8841",
    operator: "Auto-AOI-01"
  },
  {
    id: "#PCB-2025-0840",
    defectType: "Open",
    defectKey: "open_circuit",
    color: "#ef4444",
    badgeBg: "#fee2e2",
    confidence: 94.8,
    timestamp: "14:19:30",
    timeAgo: "13m ago",
    status: "Defect Found",
    statusType: "defect",
    thumbnail: "/prediction_samples/pred_01_open_circuit_12_jpg.rf.640be935ceb2f18fbee221d8cf56bad3.jpg",
    details: "Break in power distribution trace on main DC bus rail",
    layer: "Bottom Ground Plane",
    lot: "Lot B4-8840",
    operator: "Auto-AOI-02"
  },
  {
    id: "#PCB-2025-0839",
    defectType: "Passed (None)",
    defectKey: "passed",
    color: "#10b981",
    badgeBg: "#d1fae5",
    confidence: 99.5,
    timestamp: "14:15:02",
    timeAgo: "17m ago",
    status: "Passed",
    statusType: "passed",
    thumbnail: "/prediction_samples/pred_pku_pcb_0006.jpg",
    details: "Zero anomalies detected across all 4 inspection sectors",
    layer: "All Layers Compliant",
    lot: "Lot B4-8839",
    operator: "Auto-AOI-01"
  },
  {
    id: "#PCB-2025-0838",
    defectType: "Spur",
    defectKey: "spur",
    color: "#8b5cf6",
    badgeBg: "#ede9fe",
    confidence: 91.7,
    timestamp: "14:08:44",
    timeAgo: "24m ago",
    status: "Review Req.",
    statusType: "review",
    thumbnail: "/prediction_samples/pred_01_spur_05_jpg.rf.1ff0881213a4714603198bb630a16247.jpg",
    details: "Copper spur outgrowth extending 0.08mm toward ground plane",
    layer: "Layer 1 Routing",
    lot: "Lot B4-8838",
    operator: "Auto-AOI-01"
  },
  {
    id: "#PCB-2025-0837",
    defectType: "Missing Hole",
    defectKey: "missing_hole",
    color: "#3b82f6",
    badgeBg: "#dbeafe",
    confidence: 97.1,
    timestamp: "14:02:18",
    timeAgo: "30m ago",
    status: "Defect Found",
    statusType: "defect",
    thumbnail: "/prediction_samples/pred_01_missing_hole_03_jpg.rf.2fe59ac9823cde169d3e2049886eaf6a.jpg",
    details: "Via drill missing at coordinate X:241.2, Y:118.4",
    layer: "Drill Through-Hole",
    lot: "Lot B4-8837",
    operator: "Auto-AOI-03"
  },
  {
    id: "#PCB-2025-0836",
    defectType: "Spurious Copper",
    defectKey: "spurious_copper",
    color: "#14b8a6",
    badgeBg: "#ccfbf1",
    confidence: 89.4,
    timestamp: "13:55:40",
    timeAgo: "37m ago",
    status: "Review Req.",
    statusType: "review",
    thumbnail: "/prediction_samples/pred_01_spurious_copper_02_jpg.rf.0a382538d512ca6797cbf5791185a974.jpg",
    details: "Isolated unetched copper island detected in dielectric spacing",
    layer: "Layer 3 Signal",
    lot: "Lot B4-8836",
    operator: "Auto-AOI-02"
  },
  {
    id: "#PCB-2025-0835",
    defectType: "Passed (None)",
    defectKey: "passed",
    color: "#10b981",
    badgeBg: "#d1fae5",
    confidence: 99.8,
    timestamp: "13:49:10",
    timeAgo: "43m ago",
    status: "Passed",
    statusType: "passed",
    thumbnail: "/prediction_samples/pred_pku_pcb_0010.jpg",
    details: "High-density interconnect pads verified with zero solder defects",
    layer: "All Layers Compliant",
    lot: "Lot B4-8835",
    operator: "Auto-AOI-01"
  }
];

export const recentAlerts = [
  {
    id: "alert-01",
    type: "CRITICAL SHORT",
    severity: "critical",
    timeAgo: "4 mins ago",
    description: "Trace bridge on Layer 2 SMT pad",
    boardId: "#PCB-2025-0842",
    hasAction: true,
    actionText: "Review Anomaly",
    inspectionRef: "#PCB-2025-0842"
  },
  {
    id: "alert-02",
    type: "MULTIPLE DEFECTS",
    severity: "warning",
    timeAgo: "24 mins ago",
    description: "Open circuit + Spur detected simultaneously",
    boardId: "#PCB-2025-0838",
    hasAction: false,
    inspectionRef: "#PCB-2025-0838"
  },
  {
    id: "alert-03",
    type: "PROCESS ANOMALY",
    severity: "info",
    timeAgo: "1 hour ago",
    description: "3 consecutive boards exhibiting open solder traces",
    location: "Line 2 AOI Station",
    hasAction: false,
    inspectionRef: "#PCB-2025-0840"
  }
];

export const pipelineHealth = [
  { label: "API Core", value: "FastAPI v0.110.0 (24ms)", status: "online" },
  { label: "AI Detector", value: "YOLOv8-PCB (CUDA ON)", status: "online" },
  { label: "Last Sync", value: "14:28:12 UTC (Line 1)", status: "normal" },
  { label: "Telemetry", value: "Synchronized (30s)", status: "synced" }
];

// Compatibility aliases
export const mockDetections = recentInspections;
export const alerts = recentAlerts;

