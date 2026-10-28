# PCB model accuracy: diagnosis and recovery plan

Scope: all six existing labels. This is a targeted audit of the running project, not a promise that a larger model will solve the problem. No production checkpoint has been replaced or retrained. The new full recovery run still needs execution on Kaggle.

## A. Current model diagnosis

The task is bounding-box object detection: missing_hole, mouse_bite, open_circuit, short, spur, spurious_copper. A break through a copper conductor belongs to open_circuit. A crack in the board substrate, solder joint or component is not automatically one of these six labels.

**The main failure is cross-domain generalization, plus incomplete class coverage in the current target specialist.** The source YOLOv8s was trained on 480 large PKU board images, with 120 validation images from separate board-template groups. The target YOLOv8s was fine-tuned on 3,606 mostly 226×226 DsPCBSD+ close-ups, with 403 validation images. The target training contains no missing_hole labels; its deployment explicitly filters class 0 out. Removing that filter would not restore missing-hole training.

The target training did not replay source examples. Both networks recognize their own image distribution much better than the other distribution. Source after target fine-tuning had not previously been evaluated on PKU; that gap is now checked on a fixed validation sample. This is consistent with lost source-domain capability, but the audit cannot attribute the entire gap to forgetting rather than appearance/scale shift.

### Measurements from this audit

`diagnosis.json` contains predictions and same-class, one-to-one IoU≥0.50 matching. Images were selected with seed 42 from validation only: 3 per source class (18 source images), 6 per target class (30 target images). Precision/recall below are micro counts at confidence 0.25, not mAP and not independent final-test results.

- Source on source: 70/84 labelled defects found, 27 false detections; recall 83.33%, precision 72.16%.
- Target on source: 0/84 found, 936 false detections.
- Source on target: 0/100 found and no predictions above 0.25.
- Target on target: 81/100 found, 41 false detections; recall 81.00%, precision 66.39%.
- Source open_circuit: 10/11 found on the sampled source images. Target open_circuit: 15/16 found on sampled target close-ups. Obvious trace breaks can therefore be learned; choosing the wrong specialist is a concrete cause of misses.
- Source missing_hole: only 6/15 found at this threshold. Switching to source restores class coverage, not perfect missing-hole recall.

Threshold changes do not repair the domain gap. On target close-ups, lowering confidence from 0.25 to 0.10 raised recall from 81% to 91%, while precision fell from 66.39% to 46.19%. Raising it to 0.50 reduced recall to 48%. The target still found no source defects at any of those three thresholds.

A separate bounded slicing experiment used six source validation images, 512px tiles, about 20% overlap and class-aware NMS. Across 216 tiles, the target produced no correct matches at IoU 0.5/confidence 0.25 and 1,522 false detections. This rejects that specific shortcut for the current weights. It does not prove that every slicing configuration or a slice-trained model will fail.

### Existing saved benchmark results

Target development mAP50 is 78.82%, mAP50:95 37.66%, macro precision 73.01%, macro recall 75.51%. Spur development recall is 53.20%. These already indicate misses and weak localization. Source holdout mAP50 is 71.14%, mAP50:95 35.30%. Source validation was about 92.13% mAP50. These are different datasets, not a controlled source/target comparison. The development holdouts have already been inspected; none is a new untouched field test.

### Training dynamics

The 80-epoch target CSV is complete, including the resumed epochs. Best validation mAP50:95 was around epoch 61. Training box loss fell from 2.40 to 1.48; validation box loss fell from 2.21 to about 1.80 and then largely plateaued. Training classification loss fell from 1.99 to 0.93; validation reached about 1.14. There is a modest late generalization gap, not evidence that the whole run failed to optimize. More epochs alone are low priority. Gradients, calibration and multi-seed variability were not recorded, so conclusions about their stability remain unverified.

## B. Complete pipeline review

### Data collection, labels and splits

All prepared labels were checked for valid class IDs, finite positive boxes and normalized bounds. There were no invalid prepared labels or exact cross-split image hashes in either dataset. This is a structural audit, not a human verification of every annotation. Earlier raw COCO edge deviations were below 0.02 pixel and were clipped during preparation; those tiny roundoff errors do not explain the observed cross-domain collapse.

The source has 325/313/319/330/314/321 training instances across the six classes: reasonably balanced. The target has mouse_bite 1,400, open_circuit 1,053, short 563, spur 2,785 and spurious_copper 918. Missing-hole coverage is zero. Both original training sets contain zero empty-background images. Their false-positive behavior should be tested on verified normal regions and eventually complete healthy boards.

Six source training board-template groups are separated from two validation groups; test groups remain excluded. Target preparation excluded close cross-split perceptual hashes, but board identities are unavailable. Perceptual deduplication cannot establish board-level independence. Rotations and derived crops must inherit their parent's split and must never be counted as new independent boards.

Prioritize a small labelled acceptance set of the actual expected input style, including all six classes and normal boards. If only public training data is permitted, keep that rule; a realistic acceptance benchmark is still needed before claiming deployment accuracy. Inspect semantic label consistency manually, especially thin cracks, spur versus copper, and the actual meaning of missing_hole.

The official [DsPCBSD+ release](https://figshare.com/articles/dataset/DsPCBSD_/24970329) describes 10,259 images and nine categories; the project uses the five genuinely shared categories. [DeepPCB](https://github.com/tangsanli5201/DeepPCB) offers 1,500 aligned image pairs, but its pin-hole class is not a missing drilled hole. Only the other five matching categories could be mapped without a new semantic review. DeepPCB also uses a different evaluation protocol; do not transfer its benchmark scores directly. No new external dataset was downloaded for this recovery.

### Preprocessing and feature handling

The API's RGB decode → BGR NumPy input → Ultralytics preprocessing matches the documented input contract. Boxes are returned in original-image pixels. Checkpoints and class order are verified. No evidence of a channel swap, double normalization or frontend-only inference bug was found. [Ultralytics predict documentation](https://docs.ultralytics.com/modes/predict/).

Native median short box sides are 54 pixels in source training and about 12 pixels in target training. Downscaling a 3,000px board to 640px can severely shrink a thin break. Target close-ups are enlarged instead. Retain native images, use aspect-preserving letterboxing, and test training with both complete boards and accurately labelled source crops. Avoid arbitrary contrast/thresholding that removes copper detail. Blur, denoising and super-resolution are not free accuracy improvements: a thin gap can disappear or be hallucinated. Registration/template subtraction is a separate optional path only when a correctly aligned defect-free template exists.

PCA, tabular feature engineering and manual feature normalization are not first-line changes to this end-to-end detector. Edge maps may be an ablation later; changing input channels requires matching retraining and deployment.

### Augmentation

Keep mild flips, ±5° rotation, 3% translation and 10% scale changes for the first recovery. Retain gentle brightness/saturation changes. Crop labels must be recomputed; crops with badly truncated labelled objects are rejected. Validation stays unaugmented. Random erasing, strong blur and aggressive downscaling can destroy the very defect being labelled. Mosaic and MixUp remain off for this repair. Copy-Paste would need credible copper connectivity and ideally masks, not arbitrary rectangular pastes. RandAugment/AutoAugment/AugMix are later robustness experiments with severity limits and box-consistent geometry.

### Architecture and pretrained weights

Keep YOLOv8s first. It already learns the in-domain task, and changing architecture would confound the more urgent data/coverage repair. The local six-class model has about 11.13M parameters. Official 80-class YOLOv8 benchmarks list n/s/m at 3.2M/11.2M/25.9M parameters and 37.3/44.9/50.2 COCO mAP50:95, respectively. Their listed A100 TensorRT times are not this Mac or Pi's latency. Larger capacity costs memory and does not establish PCB generalization. [Official benchmark](https://docs.ultralytics.com/models/yolov8/).

A P2/smaller-stride head is a relevant later small-object ablation, with more feature-map memory. RT-DETR is a supported alternative if the repaired YOLO still fails, but its GPU benchmark is not an edge guarantee. DINOv2/contrastive pretraining can help representations but requires a detection adaptation stage. Segmentation with Dice/IoU losses needs pixel masks; bounding boxes alone do not justify claiming crack segmentation. Distill a validated stronger teacher into a smaller edge model only after teacher accuracy is demonstrated. Do not treat a generic COCO or vision-language checkpoint as an already trained six-class PCB detector.

### Losses, optimization and search

Pinned Ultralytics YOLOv8 detection uses BCE classification and box/DFL regression components. DFL is already present; proposing it as a new modification would be misleading. Keep the existing AdamW/cosine/warmup recipe for the data repair. Use AMP on T4 and effective batch 32 through accumulation. Neither a bigger batch nor SGD/AdamW swapping is a substitute for missing examples. Gradient clipping is useful only if measured spikes occur. EMA/checkpoint selection should remain part of the standard trainer. [Pinned loss implementation](https://github.com/ultralytics/ultralytics/blob/v8.4.142/ultralytics/utils/loss.py).

After repairing coverage, compare class-aware sampling or supported class weighting for missing-hole/short examples. Focal-style losses can emphasize hard cases but also mislabeled cases; changing detector internals is a later controlled test. Label smoothing risks suppressing already weak thin-defect confidence. Contrastive/domain/MMD losses remain experimental here; the existing MMD experiment did not improve results.

Do not launch grid search or population-based training under the hours-level constraint. After the first recovery, a small random/Optuna search over learning rate, source replay ratio and image resolution is more useful. ASHA/Hyperband pruning requires enough epochs to recognize slower learning; stopping at epoch 5 could discard a model that peaks around epoch 61. Bayesian tuning is worthwhile only with a trusted fixed validation objective and enough trials. Multiple seeds and board-group cross-validation are later confirmation, not substitutes for an untouched acceptance set.

### Advanced training and inference

Mine false positives/negatives from validation for diagnosis; put newly annotated training examples into a separate training pool, never move validation images into training. Active learning should prioritize uncertain, diverse and mismatched examples. Do not auto-pseudo-label screenshots with the current cross-domain teacher: its measured failures would be recycled as truth. Soft Teacher/self-training requires a useful in-domain teacher and unlabelled data. Curriculum learning, feature distillation and multitask/domain alignment are lower priority than mixed supervised replay. Online test-time adaptation can drift and contaminate a benchmark; keep it off.

Test TTA, larger input, class-aware NMS, Soft-NMS and ensembling separately on validation. Weighted fusion cannot create evidence when both models miss the object. Tiling costs extra forward passes and has already failed in one relevant local diagnostic. Calibrate class-specific decision thresholds on validation and freeze them before final evaluation. Confidence is not a calibrated probability of correctness. Keep UI/API/ONNX preprocessing and thresholds identical and check exported predictions against PyTorch before Pi deployment.

## C. Verified paper review

Literature results below are from the cited tasks, not expected gains on this project's data. No universal PCB percentage improvement is claimed.

1. **Slicing Aided Hyper Inference and Fine-tuning for Small Object Detection (2022), Akyon, Altinuc, Temizel.** Tile inputs to preserve small-object scale and merge predictions. The authors report AP increases of 6.8%, 5.1% and 5.3% for FCOS/VFNet/TOOD on aerial benchmarks, with further gains from slice-aware fine-tuning. Relevant to large boards, with inference cost proportional to tile count and boundary/duplicate risks. Implementation moderate. Our current target weights failed the tested slice configuration, so this is not deployed. [Paper](https://arxiv.org/abs/2202.06934).
2. **TIDE: A General Toolbox for Identifying Object Detection Errors (2020), Bolya, Foley, Hays, Hoffman.** Separates error contributions instead of hiding them in one mAP. Demonstrated across four datasets and seven recognition models; it is an analysis tool rather than an AP-gain method. High relevance to misses versus wrong-class versus inaccurate boxes. Low/moderate offline integration cost; needs correct GT/export conventions. [Paper](https://arxiv.org/abs/2008.08115).
3. **Feature Pyramid Networks for Object Detection (2017), Lin et al.** Top-down/lateral feature fusion improves multi-scale features on COCO with marginal feature-extractor overhead. Supports testing high-resolution heads, not inventing an extra FPN as if YOLO lacked multiscale processing. A new P2 head has meaningful memory cost and requires retraining. Moderate/high implementation effort; no transferable PCB gain assumed. [Paper](https://arxiv.org/abs/1612.03144).
4. **Focal Loss for Dense Object Detection (2017), Lin, Goyal, Girshick, He, Dollár.** Downweights easy classification examples. RetinaNet reported 37.8 COCO AP for its ResNet-101/800 configuration in the paper's comparison. Relevant to hard negatives after annotation review, but noisy annotations may be amplified. Low arithmetic overhead, moderate integration/testing cost in the current YOLO loss. [Paper](https://arxiv.org/abs/1708.02002).
5. **Generalized Focal Loss (2020), Li et al.** Models localization distributions and quality-aware scores; reported 45.0 COCO AP with ResNet-101 under its study setup. Distributional box regression is already part of this YOLO implementation. Relevant as background for the localization gap, not a new checkbox fix. A wholesale loss replacement adds high testing cost. [Paper](https://arxiv.org/abs/2006.04388).
6. **iCaRL: Incremental Classifier and Representation Learning (2017), Rebuffi et al.** Retains exemplars and representations to learn incremental classes; evaluated on CIFAR-100 and ImageNet. Supports the motivation for retaining earlier source examples, but this is classification evidence rather than a guarantee for PCB detection replay. Mixed replay is straightforward; exact iCaRL adaptation is unnecessary here. [Paper](https://arxiv.org/abs/1611.07725).
7. **AugMix (2020), Hendrycks et al.** Mixes augmentation chains and consistency objectives to improve classification corruption robustness; authors report substantially reduced robustness/uncertainty gaps. Relevant to illumination robustness, with extra processing and detection-label adaptation. Do not transfer its classification gains to microscopic defects or apply transformations that erase gaps. [Paper](https://arxiv.org/abs/1912.02781).
8. **Simple Copy-Paste is a Strong Data Augmentation Method for Instance Segmentation (2021), Ghiasi et al.** Pastes segmented objects; reports +0.6 mask AP/+1.5 box AP over the previous COCO state of the art and rare-category gains on LVIS. PCB connectivity and masks make faithful adaptation harder than ordinary objects. Moderate/high annotation/engineering cost; deferred. [Paper](https://arxiv.org/abs/2012.07177).
9. **End-to-End Semi-Supervised Object Detection with Soft Teacher (2021), Xu et al.** Confidence-weighted teacher pseudo-labels and box selection; reported +3.6 COCO mAP using extra unlabelled COCO images. Relevant only after establishing a reliable in-domain teacher. Adds teacher computation and implementation complexity; current cross-domain predictions are too unreliable for unattended pseudo-labeling. [Paper](https://arxiv.org/abs/2106.09018).
10. **DETRs Beat YOLOs on Real-time Object Detection (2024), Zhao et al.** RT-DETR uses a hybrid encoder/query selection without conventional NMS. Reported R50/R101 53.1/54.3 COCO AP and 108/74 FPS on T4 in the authors' setup. Strong open-source alternative, not measured PCB/Pi performance. More integration/memory risk than retaining this working YOLOv8s. [Paper](https://arxiv.org/abs/2304.08069).
11. **DINOv2 (2023/2024), Oquab et al.** Curated large-scale self-supervised visual pretraining, with improvements across image/pixel transfer benchmarks. It provides representations rather than this six-label detector. Downloading a smaller pretrained model is feasible; detection fine-tuning and deployment validation remain substantial work. Low priority under the current time budget. [Paper](https://arxiv.org/abs/2304.07193).
12. **Soft-NMS (2017), Bodla et al.** Decays overlapping detection scores; reported about 1.1–1.3 COCO mAP improvement for the studied detectors without retraining. Low/moderate implementation cost. Relevant if good overlapping candidates are being suppressed, not if no correct candidate exists. Evaluate duplicate/false-positive tradeoffs. [Paper](https://arxiv.org/abs/1704.04503).
13. **On Calibration of Modern Neural Networks (2017), Guo et al.** Temperature scaling improves calibration in image/document classification studies. Calibration changes interpretation/threshold selection, not missing visual features. Detection-aware calibration requires held-out matched/unmatched predictions. Small runtime overhead, moderate evaluation effort. [Paper](https://arxiv.org/abs/1706.04599).
14. **Random Search for Hyper-Parameter Optimization (2012), Bergstra and Bengio.** Random search beat grid allocation under comparable budgets in the studied neural-network tasks. Supports a small targeted search, not an accuracy prediction for PCB. ASHA, Li et al. (2020), adds asynchronous early stopping for parallel tuning; population-based training, Jaderberg et al. (2017), adds adaptive population schedules. Those extra trials require resources unavailable to the immediate repair. [Random search](https://jmlr.org/papers/v13/bergstra12a.html), [ASHA](https://arxiv.org/abs/1810.05934), [PBT](https://arxiv.org/abs/1711.09846).

## D. Comparative improvement matrix

See `improvement_matrix.csv` for the ranked matrix: technique, pipeline stage, qualitative benefit, evidence, compute cost, effort and priority. No expected numeric gains are assigned without project experiments. Critical corrections precede architecture, loss and search experiments.

## E. Top ten actions for this project

1. Correct specialist selection and stop presenting the five-class target as a universal six-class model. Implemented: source is now the full-board/all-six default; target is labelled as a five-class close-up specialist.
2. Run validation on both image distributions. Implemented: fixed diagnostic predictions, GT matching and saved evidence.
3. Restore source/missing-hole examples during recovery training; do not continue target-only fine-tuning.
4. Train on both complete source boards and source training-only defect crops so the network sees both scales.
5. Add explicitly labelled-box-free source crops to expose normal textures. A full healthy-board acceptance set is still needed.
6. Preserve parent/board grouping; report source and target validation separately. Never count crops as independent boards.
7. Check missing-hole and spur performance separately rather than trusting overall mAP50.
8. Tune threshold only after recording per-class recall and false positives. A single higher threshold is inappropriate as a recall fix.
9. Collect/label representative failures within the six-label definition; use verified hard negatives and diverse boards rather than more copies of the same board.
10. Promote a checkpoint only after passing both domain gates and an actual-use acceptance set. Retain rollback weights and validate export parity.

## F. Experiment roadmap

**E0 — audited routing correction (done).** Hypothesis: specialist mismatch explains many misses. Modification: run preserved source/target at their recorded sizes on identical validation samples, keep GT and confidence matching fixed. CPU only. Measure TP/FP/FN by class/domain. Success: strong within-domain versus cross-domain separation. Observed. Failure would have directed inspection toward preprocessing/label mapping. The experiment demonstrates a failure mode, not field accuracy.

**E1 — slicing shortcut (done, rejected for deployment).** Hypothesis: smaller field of view alone rescues target on source boards. Six preselected validation images, 512px/20%-overlap slices, same target weights and thresholds. Measure per-class recall, false positives and forward-pass cost. Observed zero correct matches at IoU 0.5/0.25 and many false detections. Next: fix training distribution rather than silently enabling more slices.

**E2 — one mixed-domain recovery candidate (prepared; full run not executed).** Hypothesis: supervised replay plus training views at both scales can preserve both distributions and all six labels. Start from the preserved source checkpoint; keep architecture/loss/optimizer family stable. Train 6,485 views, validate 523 originals. One T4, 960px, batch 8/effective 32, up to 80 epochs with patience 15. This is a combined engineering repair, not an ablation isolating crop/replay effects. Compare against each specialist on its own validation domain. Review gates: source and target mAP50 and mAP50:95 may each drop at most two percentage points; source missing-hole AP50 has the same protection. Passing permits review, not automatic deployment. Failing either domain retains existing production weights.

**E3 — missing-hole/short sampling ablation.** If E2 fails rare-class checks, change only training sampling/weighting at fixed data, architecture and optimizer. Compare all-class AP, missing-hole recall, critical-defect recall and background FP/image. Reject if other-domain degradation exceeds the gate. No test-set tuning.

**E4 — scale/architecture ablation.** If residual errors are mostly small-object misses, compare 640 versus 960 or crop-aware inference first. Only then test a P2 YOLO variant or YOLOv8m/RT-DETR under the same split and budget. Record memory, latency, model size and AP50:95; do not choose on COCO marketing numbers.

**E5 — robustness and confirmation.** Test one augmentation/hard-negative intervention at a time, then repeat the most important winning comparison with at least three seeds if time permits. Use board-level resampling/confidence intervals where board identity exists. Freeze all choices before a new independent acceptance set. Explicitly report uncertainty if resources prevent repeated seeds.

## G. Recommended final pipeline

Input scope and image quality check → consistent RGB/BGR decode/letterbox → one validated six-class mixed-domain detector → optional validated scale policy → class-aware NMS → validation-calibrated thresholds → saved annotations/actual class coverage → defect-specific review and maintenance history. Keep uncertain/no-detection results distinct from verified good boards. Maintenance prediction remains a separate, currently simulated component.

Use a known six-class PCB checkpoint as the warm start, not a random network. Prefer YOLOv8s until controlled evidence supports another architecture. General-purpose pretrained open-source models still require PCB fine-tuning. No verified drop-in checkpoint has been shown here to solve these exact six labels across both domains.

## H. Implementation and files

- `accuracy/audit.py`: complete prepared-data structure audit and fixed validation-sample inference.
- `accuracy/check_tiles.py`: bounded slicing diagnostic. Neither script evaluates test images for tuning.
- `accuracy/build_recovery.py`: mixed-domain package and source-only derived crops, hashes and provenance. Package has 6,485 training views and 523 validation originals. Zero test images are included.
- `accuracy/train_recovery.py`: one resumable full-model run, domain-specific baseline/candidate evaluation, protected class coverage and export gates.
- `accuracy/PCB_Six_Class_Recovery_Kaggle.ipynb`: upload package, select T4, Run All. Uses bounded dataset discovery, progress logs and local hashing to avoid silent remote directory scans.
- `accuracy/pcb_six_class_recovery.zip`: about 1,024 MiB; preserves original JPEGs, labels, checkpoints and training script.
- `reports/accuracy/recovery_preflight.json`: precise local functional-check scope. CPU160 smoke training is not the actual accuracy experiment.

Download the checkpoint ZIP during training. Kaggle temporary storage is not a backup. Resume uses optimizer state and the same data/recipe signature. Completed runs are reused. The new candidate is never copied over `phase1_final/model.pt` or `phase2/current_target/model.pt` automatically.

## I. Remaining uncertainties and limits

We have not received the raw missed image with verified defect boxes, so its exact missed features cannot yet be scored. The earlier screenshot is a populated, blurry board, which adds another unmeasured shift. We have not established six-class reliability on assembled boards, hidden/internal cracks or substrate damage. We have not manually adjudicated every label, established target board identities, measured negative full-board specificity, calibrated probabilities, run the full recovery, validated its T4 peak memory, or measured Pi speed. No training recipe can promise the best result in one run. The evidence supports fixing class coverage and domain training before spending time on a new architecture.
