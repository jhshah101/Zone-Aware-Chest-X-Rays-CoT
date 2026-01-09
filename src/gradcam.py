from typing import Tuple
import numpy as np
import cv2
import torch
import torch.nn.functional as F

from .model import ResNet18Backbone

class GradCAM:
    def __init__(self, backbone: ResNet18Backbone):
        self.backbone = backbone
        self.gradients = None
        self.activations = None

        last_module = list(self.backbone.stem.children())[-1]
        last_module.register_forward_hook(self._fwd)
        last_module.register_full_backward_hook(self._bwd)

    def _fwd(self, module, inp, out):
        self.activations = out

    def _bwd(self, module, grad_in, grad_out):
        self.gradients = grad_out[0]

    def compute(self):
        grads = self.gradients
        acts = self.activations
        weights = grads.mean(dim=(2,3), keepdim=True)
        cam = (weights * acts).sum(dim=1)
        cam = F.relu(cam)

        cam_flat = cam.view(cam.size(0), -1)
        cam_min = cam_flat.min(dim=1)[0].view(-1,1,1)
        cam_max = cam_flat.max(dim=1)[0].view(-1,1,1)
        cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)
        return cam.detach()

def overlay_cam_on_crop(crop_rgb: np.ndarray, cam_2d: np.ndarray, alpha: float = 0.45):
    cam_resized = cv2.resize(cam_2d, (crop_rgb.shape[1], crop_rgb.shape[0]))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = (1 - alpha) * crop_rgb + alpha * heatmap
    return np.clip(overlay, 0, 255).astype(np.uint8)

def paste_crop_back(base_rgb: np.ndarray, crop_overlay: np.ndarray, box: Tuple[int,int,int,int]):
    x1,y1,x2,y2 = box
    h = y2 - y1
    w = x2 - x1
    crop_overlay = cv2.resize(crop_overlay, (w, h))
    out = base_rgb.copy()
    out[y1:y2, x1:x2] = crop_overlay
    return out
