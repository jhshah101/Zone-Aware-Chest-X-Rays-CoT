import os, glob
from typing import List, Dict, Any
import cv2
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from .yolo_lobes import detect_lobes, slots_from_lobes

def build_records(csv_path: str, dataset_dir: str, folder_map: Dict[str, str]) -> List[Dict[str, Any]]:
    df = pd.read_csv(csv_path, encoding="latin1")
    records = []
    for _, row in df.iterrows():
        case_id = str(row["Case ID"]).strip()
        label_raw = str(row["Type"]).strip()
        folder = folder_map.get(label_raw.lower(), folder_map.get("normal", "Normal"))
        folder_path = os.path.join(dataset_dir, folder)

        if not os.path.isdir(folder_path):
            candidates = [d for d in os.listdir(dataset_dir) if d.lower().startswith(folder.lower()[:4])]
            if candidates:
                folder_path = os.path.join(dataset_dir, candidates[0])

        matches = glob.glob(os.path.join(folder_path, f"{case_id}*"))
        if not matches:
            continue

        records.append({"ImagePath": matches[0], "Type": label_raw})
    return records

def encode_and_split(records: List[Dict[str, Any]], test_size: float = 0.2, seed: int = 42):
    le = LabelEncoder()
    le.fit([r["Type"] for r in records])

    for r in records:
        r["LabelIdx"] = int(le.transform([r["Type"]])[0])

    train_recs, val_recs = train_test_split(
        records,
        test_size=test_size,
        random_state=seed,
        stratify=[r["LabelIdx"] for r in records]
    )
    return train_recs, val_recs, le

class LobeDataset:
    def __init__(self, records, yolo_model, conf_th: float, num_lobes: int):
        self.records = records
        self.yolo = yolo_model
        self.conf_th = conf_th
        self.num_lobes = num_lobes

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx: int):
        rec = self.records[idx]
        path = rec["ImagePath"]
        label = rec["LabelIdx"]

        img_bgr = cv2.imread(path)
        if img_bgr is None:
            raise FileNotFoundError(path)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        lobes = detect_lobes(img_rgb, self.yolo, conf_th=self.conf_th)
        slots = slots_from_lobes(lobes, num_lobes=self.num_lobes)

        return {"path": path, "img_rgb": img_rgb, "slots": slots, "label": label}

def collate_fn(batch):
    return batch
