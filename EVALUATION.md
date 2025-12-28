# Evaluation Protocol

## Purpose
This document defines how to evaluate the effectiveness, robustness, and reliability
of Media Auth Forensics before production use.

---

## Metrics

### Primary Metrics
- ROC-AUC
- Accuracy
- Precision / Recall
- F1 Score

### Security-Critical Metrics
- False Positive Rate at high True Positive Rate (FPR@TPR)
- Worst-case adversarial score
- Cross-dataset generalization

---

## Datasets

Recommended datasets:
- FaceForensics++
- Celeb-DF
- DFDC
- Internal real-world benign data

Important:
Never evaluate on training data.
Always keep a strictly unseen holdout set.

---

## Evaluation Phases

### Phase 1: Clean Data Evaluation
- No recompression
- No resizing
- Native resolution

Purpose:
Baseline model performance.

---

### Phase 2: Real-World Transformations
Apply:
- JPEG recompression
- Scaling
- Cropping
- Color shifts

Purpose:
Assess robustness to platform pipelines.

---

### Phase 3: Adversarial Stress Testing
Apply:
- DCT low-pass filtering
- Blur-based smoothing
- Multi-stage transformations

Purpose:
Measure bypass resistance.

---

### Phase 4: Cross-Dataset Testing
Train on Dataset A, test on Dataset B.

Purpose:
Detect overfitting to dataset artifacts.

---

## Target Performance

Recommended minimum targets (guidelines, not guarantees):
- ROC-AUC ≥ 0.96 on held-out test data
- FPR ≤ 3% at TPR ≥ 95%
- Stable temporal patterns for videos

---

## Validation Rules

- Report confidence intervals
- Log all transformations applied
- Record model version and hash
- Preserve raw outputs for audit

---

## Human Review Integration

Automated detection must be:
- Advisory, not authoritative
- Reviewed by trained analysts for high-risk cases

---

End of document
