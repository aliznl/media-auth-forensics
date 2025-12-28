# Media Auth Forensics

Multimodal AI for detecting AI-generated and human-manipulated images and videos using forensic, deep learning, and adversarial-robust methods.

## What this project does
This repository provides a practical framework for media authenticity analysis that:
- Works on images and videos
- Does not require faces (supports food, documents, scenes, text)
- Produces region-level “suspicious neighborhood” maps (0/1 mask + bounding boxes)
- Applies adversarial robustness transforms to simulate bypass attempts
- Smooths video decisions over time via a Kalman filter
- Supports training and fine-tuning on FaceForensics++ frames

## Core features
- **Generic irregularity scan** (frequency + noise inconsistency) for any image
- **Optional face module** (RetinaFace) + face-crop classification
- **Adversarial suite** (JPEG, blur, DCT low-pass, crop/resize)
- **Temporal consistency** (Kalman smoothing) for video 0/1 pattern
- **Training** script for FF++ frames
- **Structured JSON outputs** for audits and integrations

## Quick start
### 1) Install
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -e .
