import os
import cv2
import torch

from .config import load_config
from .yolo_lobes import load_yolo, detect_lobes, slots_from_lobes
from .model import ZoneCoTClassifier
from .gradcam import GradCAM, overlay_cam_on_crop, paste_crop_back

def load_ckpt(model, ckpt_path: str, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["state"])
    classes = ckpt["classes"]
    return classes

@torch.no_grad()
def predict(model, classes, yolo, img_path: str, conf_th: float, num_lobes: int):
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        raise FileNotFoundError(img_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    lobes = detect_lobes(img_rgb, yolo, conf_th=conf_th)
    slots = slots_from_lobes(lobes, num_lobes=num_lobes)

    batch = [{"path": img_path, "img_rgb": img_rgb, "slots": slots, "label": 0}]
    out = model(batch, num_lobes=num_lobes, rationale_targets=None)
    logits = out["logits"]
    pred_idx = int(logits.argmax(dim=1).item())
    pred_label = classes[pred_idx]
    rationale = out["rationale"][0] if out["rationale"] is not None else None

    return pred_idx, pred_label, rationale, img_rgb, slots, out

def gradcam_overlay(model, img_rgb, slots, target_class_idx: int, out_dict, save_path: str):
    model.zero_grad(set_to_none=True)
    score = out_dict["logits"][:, target_class_idx].sum()
    score.backward(retain_graph=True)

    cam_engine = GradCAM(model.backbone)
    _ = model.backbone(out_dict["flat_imgs"])
    cams = cam_engine.compute()

    composed = img_rgb.copy()
    for lid in range(5):
        lc = slots[lid]
        if lc is None:
            continue
        cam2d = cams[lid].detach().cpu().numpy()
        overlay = overlay_cam_on_crop(lc.crop_rgb, cam2d, alpha=0.45)
        composed = paste_crop_back(composed, overlay, lc.box)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, cv2.cvtColor(composed, cv2.COLOR_RGB2BGR))
    return save_path

def main(cfg_path="configs/config.yaml", image_path=None):
    cfg = load_config(cfg_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    yolo = load_yolo(cfg.yolo_model_path)

    # dummy init; overwritten by checkpoint
    model = ZoneCoTClassifier(num_classes=4, img_size=cfg.img_size,
                              hidden_dim=cfg.hidden_dim, use_rationale=cfg.use_rationale).to(device)
    classes = load_ckpt(model, cfg.ckpt_path, device)
    model.eval()

    if image_path is None:
        raise ValueError("Provide --image path")

    pred_idx, pred_label, rationale, img_rgb, slots, out = predict(
        model, classes, yolo, image_path, cfg.conf_th, cfg.num_lobes
    )

    print("Prediction:", pred_label)
    if rationale:
        print("Rationale:", rationale)

    save_path = "outputs/gradcam_overlay.jpg"
    gradcam_overlay(model, img_rgb, slots, pred_idx, out, save_path)
    print("Saved Grad-CAM overlay:", save_path)

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--image", required=True)
    args = ap.parse_args()
    main(args.config, args.image)
