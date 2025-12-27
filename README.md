# Media Auth Forensics

Multimodal AI for detecting AI-generated and human-manipulated images and videos using forensic, deep learning, and adversarial-robust methods.

## Why this exists
Synthetic and manipulated media enables fraud, identity scams, misinformation, and security incidents. This project focuses on reliable detection under real-world conditions, including recompression, resizing, filtering, and common anti-forensic techniques.

## Core capabilities
- Image and video analysis with frame sampling
- Face-aware detection (RetinaFace) plus whole-image scanning for non-face content (food, documents, scenes)
- Region-level anomaly maps (0/1 mask + bounding boxes of suspicious neighborhoods)
- Temporal consistency modeling (Kalman smoothing across frames)
- Adversarial test harness (JPEG, blur, DCT low-pass, crop/resize) to evaluate bypass attempts
- Training pipelines (FaceForensics++ fine-tuning) and model export

## Status
Early-stage. Interfaces may change. Recommended for controlled testing and research until a stable release.

## Quick start
1) Create environment:
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
