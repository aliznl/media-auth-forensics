import os
import json
from typing import Dict, Any, Optional, List

import numpy as np
import torch
from PIL import Image

from torchvision import transforms

from media_auth_forensics.utils.io import file_sha256, safe_open_image, extract_frames, bgr_to_pil
from media_auth_forensics.detection.retinaface_wrapper import RetinaFaceWrapper
from media_auth_forensics.infer.region_scan import irregularity_scan
from media_auth_forensics.infer.temporal import kalman_binary_pattern
from media_auth_forensics.infer.adversarial import generate_variants
from media_auth_forensics.models.xception_forensics import ForensicsBinaryClassifier
from media_auth_forensics.models.model_identifier import GeneratorIdentifier


def _load_detection_model(checkpoint_path: str, device: str) -> ForensicsBinaryClassifier:
    """
    Load the binary forensics detector from a checkpoint.
    The checkpoint is expected to have {"model_state": state_dict}.
    """
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    model = ForensicsBinaryClassifier(backbone="efficientnet_b3", pretrained=False)
    ckpt = torch.load(checkpoint_path, map_location=dev)
    model.load_state_dict(ckpt["model_state"])
    model.to(dev).eval()
    return model


def _load_identifier_model(checkpoint_path: str, device: str, num_classes: int = 8) -> GeneratorIdentifier:
    """
    Load the generator/model-family identifier.
    Checkpoint is expected to have {"model_state": state_dict}.
    """
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    model = GeneratorIdentifier(num_classes=num_classes, backbone="resnet18", pretrained=False)
    ckpt = torch.load(checkpoint_path, map_location=dev)
    model.load_state_dict(ckpt["model_state"])
    model.to(dev).eval()
    return model


def _torch_preprocess(size: int = 224) -> transforms.Compose:
    """
    Torch preprocessing aligned with ImageNet-style backbones.
    """
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def _prob_from_logits(logits: torch.Tensor) -> float:
    """
    Convert a single logit tensor to probability.
    """
    return float(torch.sigmoid(logits).detach().cpu().numpy().squeeze())


def infer_path(
    input_path: str,
    detection_checkpoint: Optional[str] = None,
    id_checkpoint: Optional[str] = None,
    device: str = "cpu",
    max_frames: int = 120,
    stride: int = 5,
    run_adversarial_variants: bool = True,
) -> Dict[str, Any]:
    """
    Main app API:
      - Accept image or video path
      - Output JSON report:
          * global suspicion
          * region irregularities (0/1 neighborhoods)
          * face-based classifier results (if face detector + checkpoint)
          * generator/model-family probabilities (if identifier checkpoint)
          * Kalman-smoothed 0/1 temporal pattern for videos
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(input_path)

    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    preprocess = _torch_preprocess()

    # Optional learned models
    det_model = _load_detection_model(detection_checkpoint, device) if detection_checkpoint else None
    id_model = _load_identifier_model(id_checkpoint, device) if id_checkpoint else None

    # Face detector (optional; if unavailable it returns empty list and pipeline still works)
    face_detector = RetinaFaceWrapper(device=device)

    ext = os.path.splitext(input_path)[1].lower()
    is_video = ext in [".mp4", ".mov", ".avi", ".mkv", ".webm"]

    frames_bgr: List[np.ndarray] = []
    timestamps: List[float] = []

    if is_video:
        frames_bgr, timestamps = extract_frames(input_path, max_frames=max_frames, stride=stride)
    else:
        pil = safe_open_image(input_path)
        frames_bgr = [np.array(pil.convert("RGB"))[:, :, ::-1].copy()]  # RGB -> BGR
        timestamps = [0.0]

    frame_reports: List[Dict[str, Any]] = []
    worst_scores: List[float] = []

    for idx, bgr in enumerate(frames_bgr):
        pil_frame = bgr_to_pil(bgr)

        # 1) Generic non-face region scan (always on)
        scan = irregularity_scan(pil_frame)
        base_score = float(scan["image_score"])

        # 2) Adversarial robustness: compute worst-case region score across variants
        worst_variant = "orig"
        worst_score = base_score
        adv_list = []
        if run_adversarial_variants:
            for name, vimg in generate_variants(pil_frame):
                vscan = irregularity_scan(vimg)
                s = float(vscan["image_score"])
                adv_list.append({"variant": name, "score": s})
                if s > worst_score:
                    worst_score = s
                    worst_variant = name

        # 3) Face detection + learned detection model (if provided)
        faces = face_detector.detect_faces(bgr, conf_threshold=0.6)
        face_preds = []
        if det_model is not None and faces:
            for (x, y, w, h, conf) in faces:
                crop = bgr[y:y + h, x:x + w]
                if crop.size == 0:
                    continue
                crop_pil = bgr_to_pil(crop)
                x_t = preprocess(crop_pil).unsqueeze(0).to(dev)
                with torch.no_grad():
                    logit = det_model(x_t)
                face_preds.append({
                    "bbox": [int(x), int(y), int(w), int(h)],
                    "face_score": _prob_from_logits(logit),
                    "conf": float(conf),
                })

        # 4) Model-family identification (whole-frame, optional)
        model_id = None
        if id_model is not None:
            x_t = preprocess(pil_frame).unsqueeze(0).to(dev)
            with torch.no_grad():
                logits = id_model(x_t)
                probs = torch.softmax(logits, dim=1).detach().cpu().numpy().squeeze().tolist()
            model_id = {"probs": probs}

        # Combine (simple fusion): worst_score + max(face_score) if available
        learned_max = max([p["face_score"] for p in face_preds], default=0.0)
        final_frame_score = float(min(1.0, 0.55 * worst_score + 0.45 * learned_max))

        frame_reports.append({
            "frame_index": idx,
            "timestamp": float(timestamps[idx]) if idx < len(timestamps) else None,
            "region_score": base_score,
            "worst_region_score": worst_score,
            "worst_variant": worst_variant,
            "adversarial_variants": adv_list,
            "regions": scan["regions"],
            "face_predictions": face_preds,
            "model_identifier": model_id,
            "final_frame_score": final_frame_score,
        })

        worst_scores.append(final_frame_score)

    temporal_01 = kalman_binary_pattern(worst_scores, threshold=0.5) if is_video else [1 if worst_scores[0] >= 0.5 else 0]

    report = {
        "file": os.path.basename(input_path),
        "sha256": file_sha256(input_path),
        "is_video": is_video,
        "frames_analyzed": len(frame_reports),
        "final_score_max": float(max(worst_scores) if worst_scores else 0.0),
        "final_score_mean": float(sum(worst_scores) / max(1, len(worst_scores))),
        "temporal_binary_pattern": temporal_01,
        "frames": frame_reports,
    }
    return report


def adversarial_folder_test(input_dir: str, output_dir: str) -> None:
    """
    Run adversarial suite over all images in a folder and write one JSON per file.

    Purpose:
      Validate bypass attempts and robustness under common transforms.
    """
    os.makedirs(output_dir, exist_ok=True)

    for fn in os.listdir(input_dir):
        if not fn.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
            continue
        path = os.path.join(input_dir, fn)
        img = safe_open_image(path)

        results = []
        for name, v in generate_variants(img):
            scan = irregularity_scan(v)
            results.append({"variant": name, "score": float(scan["image_score"])})

        out_path = os.path.join(output_dir, fn + ".adv.json")
        with open(out_path, "w") as f:
            json.dump({"file": fn, "sha256": file_sha256(path), "results": results}, f, indent=2)
