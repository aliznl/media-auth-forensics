from typing import List, Tuple, Optional
import numpy as np
import cv2

try:
    # This import may differ depending on the RetinaFace implementation you choose.
    # Replace with your preferred library (insightface, retinaface, etc.).
    from retinaface import RetinaFace  # type: ignore
    RETINAFACE_AVAILABLE = True
except Exception:
    RETINAFACE_AVAILABLE = False


class RetinaFaceWrapper:
    """
    Face detector wrapper:
      - Primary: RetinaFace (if installed)
      - Fallback: OpenCV DNN face detector (if model files provided)
      - Safe behavior: if neither available, returns empty list (non-face pipeline still works)
    """

    def __init__(
        self,
        device: str = "cpu",
        opencv_proto_path: Optional[str] = None,
        opencv_model_path: Optional[str] = None,
    ):
        """
        Initialize the detector.

        Args:
          device: 'cpu' or 'cuda' (depends on chosen RetinaFace implementation)
          opencv_proto_path: optional path to deploy.prototxt (OpenCV DNN fallback)
          opencv_model_path: optional path to caffemodel (OpenCV DNN fallback)
        """
        self.device = device
        self.retina = None
        self.cv_net = None

        if RETINAFACE_AVAILABLE:
            # NOTE: APIs differ across libraries. Adjust as needed.
            # Some libs use RetinaFace.detect_faces(image) returning dict of faces.
            self.retina = RetinaFace
        else:
            if opencv_proto_path and opencv_model_path:
                self.cv_net = cv2.dnn.readNetFromCaffe(opencv_proto_path, opencv_model_path)

    def detect_faces(self, bgr: np.ndarray, conf_threshold: float = 0.6) -> List[Tuple[int, int, int, int, float]]:
        """
        Detect faces in a BGR image.

        Returns:
          List of (x, y, w, h, confidence)
        """
        h, w = bgr.shape[:2]

        # RetinaFace path
        if self.retina is not None:
            faces = []
            try:
                # Typical RetinaFace API: dict keyed by face id, values contain "facial_area" and "score".
                detections = self.retina.detect_faces(bgr)
                for _, v in detections.items():
                    score = float(v.get("score", 0.0))
                    if score < conf_threshold:
                        continue
                    x1, y1, x2, y2 = v["facial_area"]
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w, x2), min(h, y2)
                    faces.append((x1, y1, max(0, x2 - x1), max(0, y2 - y1), score))
                return faces
            except Exception:
                # If RetinaFace failed for any reason, fall back silently to other methods.
                pass

        # OpenCV DNN fallback path
        if self.cv_net is not None:
            blob = cv2.dnn.blobFromImage(cv2.resize(bgr, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
            self.cv_net.setInput(blob)
            detections = self.cv_net.forward()
            faces = []
            for i in range(detections.shape[2]):
                conf = float(detections[0, 0, i, 2])
                if conf > conf_threshold:
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    x1, y1, x2, y2 = box.astype(int)
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w, x2), min(h, y2)
                    faces.append((x1, y1, max(0, x2 - x1), max(0, y2 - y1), conf))
            return faces

        # No detector available
        return []
