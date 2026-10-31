# Adapter research protocol — implementation in progress

## Research question

Can lightweight defect-region alignment improve target-domain detection with a fixed 100-labelled-image budget after source-domain training? Dataset domains are proxies for production configuration changes; they are not records of an actual machine reconfiguration.

## Implemented foundation

`src/models/defect_regions.py` adds identity-initialized residual 1x1 adapters and differentiable within-box grid sampling with unit-normalized pooled vectors. This is not torchvision ROIAlign. Existing `DefectAwareMMDLoss` computes RBF alignment over shared classes with optional class weights. CPU probe `scripts/probe_defect_adapter.py` uses source YOLOv8s maps 15/18/21: channels 128/256/512. The adapters have 58,336 parameters. At initialization predictions match the unchanged source network exactly; alignment gradients reach adapters while frozen base weights have no gradients. Evidence: `reports/revamp/defect_adapter_probe.json`. Random inputs prove architecture/gradient connectivity only.

The original pooled FC 2048→512→128 proposal does not directly preserve the detector's spatial feature-map shapes. The residual map adapters are an implementation refinement that must be explained in the paper. They are not yet installed in the app or backed by a GPU accuracy experiment.

## Required controlled experiments

All source-initialized arms start from the same verified PCB source checkpoint, use identical original K100 target membership, identical fixed source training replay membership, class mapping, detector settings, validation split, training schedule and seed. Replay must be present in controls as well as MMD arms. Exclude source/target validation and development-test data from training and replay. Source class 0 may enter source detection loss, but cannot enter cross-domain alignment because target lacks it.

1. Conventional full-model fine-tuning with the same replay and training budget.
2. Frozen base features, trainable detection head, no adapters: capacity/freezing control.
3. Frozen base features, trainable head plus adapters, alignment weight zero.
4. Same adapters/head and ordinary class-agnostic region MMD.
5. Same adapters/head and class-aware weighted region MMD.

Previously reported plain COCO K100 and PCB-source K100 results are contextual baselines; they do not isolate MMD benefit when replay, initialization or optimization differ.

Pooling must use the detector batch's already-transformed normalized boxes, after letterboxing/augmentation. Use all selected labelled regions; never use validation labels or target development-test labels. Pair batches with shared classes, and record actual per-class region counts and skipped alignment steps. Select class weights from training data only and keep them identical across repeat runs; do not tune from development-test scores. Log detection and alignment losses separately. Decide and record alignment coefficient before the scored run. An implementation smoke pilot must check gradients, finite losses, class correspondence and recoverable checkpoint restore before full GPU runs.

## Evaluation

Select checkpoints on 403-image target validation mAP50–95. Report mAP50 and mAP50–95, class results, pooled counts at fixed confidence, source-retention results, trainable parameters and measured adaptation time. Source retention must be evaluated on a held-out source split, not replay images. Limit model/threshold tuning to validation. The existing 992-image target set has already been used; call it development evaluation. Physical-board independence is UNKNOWN. There is no verified clean-board false-alarm benchmark. Single-seed results are preliminary; do not claim statistical superiority.

## Paper boundaries

State separately: proposed mechanism; implemented components; functional verification; GPU experiments; deployment. Do not describe ordinary COCO-to-target fine-tuning as evidence for weighted domain alignment. The best working app candidate remains COCO → K500. If MMD does not help, report the negative result and deployment choice. Never relabel 500 images as few-shot 50–100. Maintenance alerts use an illustrative rate rule; RUL and synthetic scenarios are demonstrations, not calibrated equipment-life prediction. Hardware is future scope per the agreed semester-project scope.

## Next implementation gate

Implement a serializable training integration, deterministic source replay and class-aware paired batches, full detection-plus-alignment loss wiring, and matched control configuration. Verify save/load and interrupted-run recovery before preparing the Kaggle pilot. No claim of a complete adapter trainer is made at this stage.


## Training integration update

`src/models/adaptation_training.py` now computes combined source/target detection loss plus region MMD. Modes full/head/adapter/mmd/weighted expose the planned controls. Frozen features use fixed running batch-normalization statistics; detector-head and adapter parameters train. Detection loss is normalized by combined batch size; alignment coefficient is explicit. At this stage weights default to unity: that is class-aware MMD, not demonstrated selective weighting. Non-uniform weights must be documented and derived from training-only counts in the full runner.

Checkpoint functions save lossless model and optimizer state plus Python/NumPy/Torch/CUDA RNG and reject source/config/experiment mismatches. Reconstruction requires the original verified source checkpoint and this code; these state files cannot be dropped into the app as ordinary YOLO checkpoints. Tests verify identical next optimizer update after restore in the same process. Fresh-process and GPU resume checks remain pending.

Eleven focused tests passed across integration, controls, feature pooling and existing MMD/RUL checks. `reports/revamp/adaptation_real_batch_probe.json` records one real source/K100 pair, transformed by direct square resize to 128 for a CPU plumbing check only; it is not the planned production letterbox/augmentation pipeline. No GPU accuracy result exists for this new implementation yet.

Remaining Kaggle-pilot requirements: fixed source replay membership and class-aware pairing identical across arms; full-resolution dataset transforms; schedule/early stopping/validation; serializable inference reconstruction; fresh-process checkpoint resume; record batch cursor/RNG/scheduler as needed. Do not claim complete end-to-end training readiness before these gates pass.

## Bounded pilot runner available

`scripts/adapter_pilot.py` now implements square letterboxing with transformed boxes, a fixed 100-image source replay pool, original K100 target membership, shared-class source/target pairs identical across arms, inverse-square-root target-training class weights, position-derived cosine learning rate, per-update checkpoints, and bounded validation. The pilot is deliberately only a handful of updates, without augmentation, EMA, full-run early stopping or source-retention evaluation. It is not the final experiment trainer.

`reports/revamp/adapter_fresh_process_recovery.json` verifies model and optimizer tensors agree exactly between uninterrupted and separately launched pause/resume CPU runs (three steps, 128 pixels). A real source/target update and one-image validation also completed at 640 pixels. Neither short-run score is accuracy evidence. AP uses explicit confidence-ordered greedy same-class IoU matching; this can differ from the default Ultralytics validator. All future comparison arms must use the same evaluator, including re-evaluated conventional baselines.

Kaggle code bundle: `artifacts/kaggle/adapter_pilot_v1/adapter_pilot_v1.zip`. Contains source only, with file checksums. Dataset and source weights remain in the verified phase2-v2 input; original selection remains in k500-candidate. Next gate: verify GPU memory/gradients, pause/restart recovery, validation and backup download before full scored runs.
