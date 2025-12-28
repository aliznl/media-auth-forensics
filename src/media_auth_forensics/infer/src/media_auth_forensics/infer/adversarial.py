from typing import List, Tuple
from PIL import Image
import numpy as np
import cv2
from io import BytesIO


def jpeg_recompress(pil_img: Image.Image, quality: int = 75) -> Image.Image:
    """
    JPEG recompress to simulate real-world sharing/re-encoding and anti-forensic degradation.
    """
    buf = BytesIO()
    pil_img.save(buf, format="JPEG", quality=int(quality))
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def gaussian_blur(pil_img: Image.Image, sigma: float = 1.5) -> Image.Image:
    """
    Blur to simulate smoothing anti-forensics and down-stream platform transforms.
    """
    arr = np.array(pil_img)
    out = cv2.GaussianBlur(arr, (0, 0), sigmaX=float(sigma), sigmaY=float(sigma))
    return Image.fromarray(out)


def dct_lowpass(pil_img: Image.Image, keep_ratio: float = 0.2) -> Image.Image:
    """
    DCT low-pass filtering similar in spirit to spectral smoothing attacks.

    keep_ratio:
      - 0.2 keeps only low-frequency coefficients
      - higher keeps more detail (less aggressive)
    """
    ycc = np.array(pil_img.convert("YCbCr"))
    H, W, C = ycc.shape
    out = np.zeros_like(ycc, dtype=np.float32)

    block = 8
    cutoff = max(1, int(block * float(keep_ratio)))

    for ch in range(C):
        channel = ycc[:, :, ch].astype(np.float32)
        for i in range(0, H - (H % block), block):
            for j in range(0, W - (W % block), block):
                patch = channel[i:i + block, j:j + block]
                d = cv2.dct(patch)
                mask = np.zeros_like(d)
                mask[:cutoff, :cutoff] = 1.0
                patch_back = cv2.idct(d * mask)
                out[i:i + block, j:j + block, ch] = patch_back

    out = np.clip(out, 0, 255).astype(np.uint8)
    rgb = cv2.cvtColor(out, cv2.COLOR_YCrCb2RGB)
    return Image.fromarray(rgb)


def crop_resize(pil_img: Image.Image, ratio: float = 0.9) -> Image.Image:
    """
    Crop and resize to simulate platform transforms and common editing pipelines.
    """
    w, h = pil_img.size
    cw, ch = int(w * ratio), int(h * ratio)
    cropped = pil_img.crop((0, 0, cw, ch))
    return cropped.resize((w, h))


def generate_variants(pil_img: Image.Image) -> List[Tuple[str, Image.Image]]:
    """
    Generate a set of adversarial / robustness variants.
    """
    return [
        ("orig", pil_img),
        ("jpeg_q90", jpeg_recompress(pil_img, 90)),
        ("jpeg_q70", jpeg_recompress(pil_img, 70)),
        ("blur_s1.0", gaussian_blur(pil_img, 1.0)),
        ("blur_s2.5", gaussian_blur(pil_img, 2.5)),
        ("dct_lp_0.2", dct_lowpass(pil_img, 0.2)),
        ("dct_lp_0.4", dct_lowpass(pil_img, 0.4)),
        ("crop_resize_90", crop_resize(pil_img, 0.9)),
    ]
