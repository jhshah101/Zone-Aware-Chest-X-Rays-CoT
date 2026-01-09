# Zone-Aware-Chest-X-Rays-CoT

**Pipeline:** YOLOv8 (5 lobes) → crop lobes → ResNet18 features → zone-wise "CoT" reasoning (GRU) → disease classification → Grad-CAM.

> Lobe class IDs are assumed fixed order (example): `0=RUL, 1=RML, 2=RLL, 3=LUL, 4=LLL`.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Configure paths

Edit `configs/config.yaml`:

- `csv_path`: CSV with columns **Case ID** and **Type**
- `dataset_dir`: root folder containing class subfolders
- `yolo_model_path`: your YOLO lobe detector weights (best.pt)

## Train

```bash
python -m src.train --config configs/config.yaml
```

Saves best checkpoint to `outputs/best_zone_cot_resnet18.pt`.

## Inference + Grad-CAM

```bash
python -m src.infer --config configs/config.yaml --image "data/images/Normal/example.png"
```

Outputs `outputs/gradcam_overlay.jpg`.

## Notes
- If you only want classification, set `use_rationale: false` in config.
- Keep medical data and large weights out of GitHub.
