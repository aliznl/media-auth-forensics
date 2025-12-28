from typing import Dict, Any, List
import numpy as np
import cv2
from PIL import Image


def _frequency_anomaly_map(pil_img: Image.Image, block: int = 16) -> np.ndarray:
    """
    Compute a frequency-based anomaly map (0..1) using block DCT high-frequency energy.

    Rationale:
      Many editing operations change local compression/frequency statistics.
      This works for faces and non-faces (food, text, scenery).

    Returns:
      heatmap float32 array (H, W) normalized to 0..1
    """
    y = np.array(pil_img.convert("YCbCr").split()[0]).astype(np.float32)
    H, W = y.shape
    gh, gw = H // block, W // block

    energies = np.zeros((gh, gw), dtype=np.float32)
    for i in range(gh):
        for j in range(gw):
            patch = y[i * block:(i + 1) * block, j * block:(j + 1) * block]
            d = cv2.dct(patch)
            hf = np.sum(np.abs(d[block // 2:, block // 2:]))
            total = np.sum(np.abs(d)) + 1e-9
            energies[i, j] = hf / total

    mu, sigma = float(energies.mean()), float(energies.std() + 1e-9)
    z = (energies - mu) / sigma
    z = (z - z.min()) / (z.max() - z.min() + 1e-9)
    return cv2.resize(z, (W, H), interpolation=cv2.INTER_LINEAR).astype(np.float32)


def _noise_inconsistency_map(pil_img: Image.Image, patch: int = 64) -> np.ndarray:
    """
    Compute a noise-residual inconsistency map (0..1).

    Rationale:
      Splicing/retouching/AI edits often introduce local noise level differences.

    Returns:
      heatmap float32 array (H, W) normalized 0..1
    """
    g = np.array(pil_img.convert("L")).astype(np.float32)
    residual = g - cv2.GaussianBlur(g, (7, 7), 0)

    H, W = g.shape
    gh, gw = H // patch, W // patch
    heat = np.zeros((gh, gw), dtype=np.float32)

    for i in range(gh):
        for j in range(gw):
            p = residual[i * patch:(i + 1) * patch, j * patch:(j + 1) * patch]
            heat[i, j] = float(np.std(p))

    heat = (heat - heat.min()) / (heat.max() - heat.min() + 1e-9)
    return cv2.resize(heat, (W, H), interpolation=cv2.INTER_LINEAR).astype(np.float32)


def irregularity_scan(pil_img: Image.Image) -> Dict[str, Any]:
    """
    Produce region-level irregularity detection for ANY image (face or non-face).

    Outputs:
      - fused_heatmap: float32 (H,W) in 0..1
      - binary_mask: uint8 (H,W) 0/1 "neighborhood" suspicious map
      - regions: list of bounding boxes with scores
      - image_score: average fused score

    This is intentionally model-agnostic and complements learned detectors.
    """
    freq = _frequency_anomaly_map(pil_img, block=16)
    noise = _noise_inconsistency_map(pil_img, patch=64)

    fused = np.clip(0.6 * freq + 0.4 * noise, 0.0, 1.0)

    # Otsu thresholding to produce 0/1 neighborhood map
    fused_u8 = (fused * 255).astype(np.uint8)
    _, thr = cv2.threshold(fused_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    binary = (thr > 0).astype(np.uint8)

    # Connected components to describe regions
    num, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    regions: List[Dict[str, Any]] = []
    for lab in range(1, num):
        x, y, w, h, area = stats[lab]
        mask = (labels == lab).astype(np.uint8)
        mean_score = float((fused * mask).sum() / (mask.sum() + 1e-9))
        regions.append({"bbox": [int(x), int(y), int(w), int(h)], "area": int(area), "mean_score": mean_score})

    return {
        "fused_heatmap": fused,     # float32 HxW
        "binary_mask": binary,      # uint8 HxW
        "regions": regions,
        "image_score": float(fused.mean()),
    }
