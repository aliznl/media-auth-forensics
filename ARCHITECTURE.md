# System Architecture

## Overview
Media Auth Forensics is a multimodal media integrity analysis system designed to detect
AI-generated and human-manipulated images and videos under real-world conditions.

The system combines:
- Classical forensic analysis
- Deep learning–based detection
- Adversarial robustness testing
- Temporal consistency modeling

The architecture is modular and extensible, supporting both research experimentation
and enterprise deployment.

---

## High-Level Pipeline

Input (Image / Video)
        |
        v
+------------------------+
| Input Validation       |
| - File type check      |
| - Size limits          |
| - Safe decoding        |
+------------------------+
        |
        v
+------------------------+
| Frame Sampling         |
| (video only)           |
+------------------------+
        |
        v
+------------------------+
| Region Irregularity    |
| Analysis (non-ML)      |
| - Frequency anomalies  |
| - Noise inconsistencies|
+------------------------+
        |
        v
+------------------------+
| Face Detection         |
| (RetinaFace optional)  |
+------------------------+
        |
        v
+------------------------+
| Learned Detection      |
| (CNN classifier)       |
+------------------------+
        |
        v
+------------------------+
| Adversarial Testing    |
| (JPEG, blur, DCT, etc) |
+------------------------+
        |
        v
+------------------------+
| Temporal Smoothing     |
| (Kalman Filter)        |
+------------------------+
        |
        v
Final Report (JSON)

---

## Key Components

### 1. Input Validation
Ensures safe processing by:
- Rejecting malformed files
- Enforcing size limits
- Preventing resource exhaustion

### 2. Region Irregularity Scanner
A model-agnostic module that detects suspicious neighborhoods in any image:
- Works for faces, food, documents, scenes, and text
- Outputs 0/1 binary masks and bounding boxes
- Resistant to simple adversarial tricks

### 3. Face Detection (Optional)
If faces are present:
- RetinaFace detects facial regions
- Face crops are analyzed independently
If no faces exist, the pipeline continues without failure.

### 4. Learned Detection Models
CNN-based classifiers trained on manipulated media datasets:
- Binary classification (real vs manipulated)
- Optional multi-class generator identification
- Designed to complement forensic signals

### 5. Adversarial Robustness Harness
Evaluates detection stability under:
- Recompression
- Blurring
- Spectral smoothing
- Cropping and resizing

Worst-case scores are used for final decisions.

### 6. Temporal Consistency Modeling
For video inputs:
- Frame scores are smoothed using a Kalman filter
- Produces a stable 0/1 manipulation pattern
- Reduces flicker and false positives

---

## Design Principles

- Fail-safe: no single module is required
- Defense-in-depth: multiple independent signals
- Adversary-aware: assumes bypass attempts
- Explainable outputs: region-level evidence
- Human-in-the-loop friendly

---

## Deployment Modes

- Research / offline analysis
- Enterprise batch processing
- Real-time stream processing (with batching)
- API-based inference (future)

---

## Non-Goals

- Perfect attribution or legal proof
- Replacement for human judgment
- Single-model dependency

---

End of document
