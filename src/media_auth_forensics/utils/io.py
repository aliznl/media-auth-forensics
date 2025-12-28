import os
import hashlib
from typing import List, Tuple
import cv2
import numpy as np
from PIL import Image


def file_sha256(path: str) -> str:
    """
    Compute SHA-256 of a file for integrity/audit logs.
    """
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def safe_open_image(path: str, max_pixels: int = 100_000_000) -> Image.Image:
    """
    Open and validate an image safely to mitigate parsing and resource exhaustion risks.

    Args:
      path: image file path
      max_pixels: safety cap to prevent out-of-memory processing

    Returns:
      PIL.Image in RGB mode

    Raises:
      ValueError for invalid images or oversized images
    """
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    try:
        img = Image.open(path)
        img.verify()
    except Exception as e:
        raise ValueError(f"Image verification failed: {e}")

    img = Image.open(path).convert("RGB")
    if img.width * img.height > max_pixels:
        raise ValueError("Image too large to process safely.")
    return img


def extract_frames(path: str, max_frames: int = 120, stride: int = 5) -> Tuple[List[np.ndarray], List[float]]:
    """
    Extract frames from a video with bounded work and timestamps.

    Args:
      path: video file path
      max_frames: cap number of frames extracted
      stride: sample every stride-th frame

    Returns:
      frames: list of BGR frames (OpenCV)
      timestamps: list of timestamps in seconds aligned with frames
    """
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError("Cannot open video file.")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    frames: List[np.ndarray] = []
    timestamps: List[float] = []

    idx = 0
    while len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % stride == 0:
            frames.append(frame)
            timestamps.append(idx / fps)
        idx += 1

    cap.release()
    return frames, timestamps


def bgr_to_pil(frame_bgr: np.ndarray) -> Image.Image:
    """
    Convert BGR OpenCV frame to PIL RGB image.
    """
    import cv2
    from PIL import Image
    return Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
