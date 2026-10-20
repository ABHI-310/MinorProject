/**
 * Model metrics — sourced from reports/source_model_metrics.json
 * These are the real evaluation results from the YOLOv8n model
 * trained on the PKU-Market-PCB dataset.
 */

export const overallMetrics = {
  mAP50: 0.0066,
  mAP50_95: 0.0027,
  meanPrecision: 0.2889,
  meanRecall: 0.0102,
  meanF1: 0.0198,
};

export const perClassMetrics = [
  { classId: 0, name: "missing_hole",    precision: 0.3333, recall: 0.0083, f1: 0.0163, mAP50: 0.0017, mAP50_95: 0.0008 },
  { classId: 1, name: "mouse_bite",      precision: 0.0000, recall: 0.0000, f1: 0.0000, mAP50: 0.0000, mAP50_95: 0.0000 },
  { classId: 2, name: "open_circuit",    precision: 0.0000, recall: 0.0000, f1: 0.0000, mAP50: 0.0000, mAP50_95: 0.0000 },
  { classId: 3, name: "short",           precision: 0.5000, recall: 0.0213, f1: 0.0408, mAP50: 0.0217, mAP50_95: 0.0123 },
  { classId: 4, name: "spur",            precision: 0.4000, recall: 0.0189, f1: 0.0360, mAP50: 0.0090, mAP50_95: 0.0021 },
  { classId: 5, name: "spurious_copper", precision: 0.5000, recall: 0.0130, f1: 0.0253, mAP50: 0.0075, mAP50_95: 0.0008 },
];

export const inferenceSpeed = {
  preprocess: 1.31,
  inference: 38.55,
  postprocess: 0.14,
  total: 40.0,
};

export const trainingInfo = {
  modelName: "YOLOv8n",
  device: "NVIDIA T4 GPU (Google Colab)",
  epochsCompleted: 50,
  batchSize: 16,
  imageSize: 640,
  trainingTime: "29m 49s",
};
