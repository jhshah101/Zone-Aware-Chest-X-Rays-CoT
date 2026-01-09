from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
import numpy as np
from ultralytics import YOLO

@dataclass
class LobeCrop:
    cls_id: int
    box: Tuple[int, int, int, int]  # x1,y1,x2,y2
    crop_rgb: np.ndarray

def load_yolo(model_path: str) -> YOLO:
    return YOLO(model_path)

def detect_lobes(img_rgb: np.ndarray, yolo_model: YOLO, conf_th: float = 0.25) -> List[LobeCrop]:
    """Detect lobes; keep best (largest area) per class id; return sorted by cls_id."""
    results = yolo_model(img_rgb, conf=conf_th, verbose=False)
    res = results[0]

    lobes: List[LobeCrop] = []
    if getattr(res, "boxes", None) is None or len(res.boxes) == 0:
        return lobes

    xyxy = res.boxes.xyxy.detach().cpu().numpy()
    cls = res.boxes.cls.detach().cpu().numpy().astype(int)

    for box, c in zip(xyxy, cls):
        x1, y1, x2, y2 = map(int, box[:4])
        x1 = max(0, x1); y1 = max(0, y1)
        x2 = min(img_rgb.shape[1]-1, x2); y2 = min(img_rgb.shape[0]-1, y2)
        if x2 <= x1 or y2 <= y1:
            continue
        crop = img_rgb[y1:y2, x1:x2].copy()
        lobes.append(LobeCrop(cls_id=c, box=(x1,y1,x2,y2), crop_rgb=crop))

    best: Dict[int, LobeCrop] = {}
    for lc in lobes:
        x1,y1,x2,y2 = lc.box
        area = (x2-x1) * (y2-y1)
        if lc.cls_id not in best:
            best[lc.cls_id] = lc
        else:
            bx1,by1,bx2,by2 = best[lc.cls_id].box
            barea = (bx2-bx1)*(by2-by1)
            if area > barea:
                best[lc.cls_id] = lc

    return [best[k] for k in sorted(best.keys())]

def slots_from_lobes(lobes: List[LobeCrop], num_lobes: int = 5) -> List[Optional[LobeCrop]]:
    slots: List[Optional[LobeCrop]] = [None] * num_lobes
    for lc in lobes:
        if 0 <= lc.cls_id < num_lobes:
            slots[lc.cls_id] = lc
    return slots
