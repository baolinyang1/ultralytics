from __future__ import annotations

from typing import List, Tuple
import time
import cv2
import numpy as np
from openvino.runtime import Core

from .config import Config


class PoseModel:
    """OpenVINO wrapper for static INT8 ONNX pose model."""

    def __init__(self, cfg: Config, device: str = "CPU"):
        self.cfg = cfg
        core = Core()
        model = core.read_model(cfg.model_path)
        model.reshape({model.inputs[0]: [1, 3, cfg.img_size, cfg.img_size]})
        self.compiled = core.compile_model(model, device)
        self.output_layer = self.compiled.output(0)

    def preprocess(self, frame_bgr: np.ndarray) -> np.ndarray:
        img = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.cfg.img_size, self.cfg.img_size), interpolation=cv2.INTER_LINEAR)
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))[None, ...]
        return img

    def infer(self, frame_bgr: np.ndarray) -> Tuple[List[np.ndarray], float]:
        """Returns (det_list, infer_ms). det_list items are det57 arrays."""
        t0 = time.perf_counter()
        out = self.compiled([self.preprocess(frame_bgr)])[self.output_layer]  # (1, N, 57)
        infer_ms = (time.perf_counter() - t0) * 1000.0

        preds = out[0]
        dets = preds[preds[:, 4] >= self.cfg.det_thresh]
        det_list = [dets[i] for i in range(dets.shape[0])] if dets.shape[0] > 0 else []
        return det_list, infer_ms
