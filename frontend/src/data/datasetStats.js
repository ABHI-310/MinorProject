/**
 * Dataset statistics — sourced from reports/dataset_analysis.json
 */

export const datasetSummary = {
  totalImages: 3326,
  totalAnnotations: 13948,
  avgAnnotationsPerImage: 4.19,
  commonResolution: "640×640",
};

export const classDistribution = [
  { name: "missing_hole",    instances: 2315, color: "#3b82f6" },
  { name: "mouse_bite",      instances: 2358, color: "#8b5cf6" },
  { name: "open_circuit",    instances: 2220, color: "#ef4444" },
  { name: "short",           instances: 2341, color: "#f59e0b" },
  { name: "spur",            instances: 2327, color: "#10b981" },
  { name: "spurious_copper", instances: 2387, color: "#ec4899" },
];
