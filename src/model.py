from typing import Optional, List, Dict, Any, Tuple
import torch
import torch.nn as nn
import torchvision
from torchvision.models import resnet18
from torchvision import transforms

from transformers import T5Tokenizer, T5ForConditionalGeneration

class ResNet18Backbone(nn.Module):
    def __init__(self, pretrained: bool = True):
        super().__init__()
        m = resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT if pretrained else None)
        self.stem = nn.Sequential(
            m.conv1, m.bn1, m.relu, m.maxpool,
            m.layer1, m.layer2, m.layer3, m.layer4
        )
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        feat_map = self.stem(x)
        vec = self.pool(feat_map).flatten(1)
        return vec, feat_map

def imagenet_transform(img_size: int):
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

class ZoneCoTClassifier(nn.Module):
    def __init__(self, num_classes: int, img_size: int = 224, hidden_dim: int = 256, use_rationale: bool = True):
        super().__init__()
        self.img_size = img_size
        self.tf = imagenet_transform(img_size)

        self.backbone = ResNet18Backbone(pretrained=True)
        self.feat_proj = nn.Linear(512, hidden_dim)
        self.gru = nn.GRU(input_size=hidden_dim, hidden_size=hidden_dim, batch_first=True)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

        self.use_rationale = use_rationale
        if use_rationale:
            self.tok = T5Tokenizer.from_pretrained("t5-small")
            self.t5 = T5ForConditionalGeneration.from_pretrained("t5-small")
            self.to_t5 = nn.Linear(hidden_dim, self.t5.config.d_model)

    def _build_lobe_tensor(self, batch_items: List[Dict[str, Any]], num_lobes: int, device: torch.device):
        B = len(batch_items)
        imgs = torch.zeros((B, num_lobes, 3, self.img_size, self.img_size), dtype=torch.float32)
        boxes_all: List[List[Optional[Tuple[int,int,int,int]]]] = []

        for i, item in enumerate(batch_items):
            boxes_i = []
            for lid in range(num_lobes):
                lc = item["slots"][lid]
                if lc is not None:
                    imgs[i, lid] = self.tf(lc.crop_rgb)
                    boxes_i.append(lc.box)
                else:
                    boxes_i.append(None)
            boxes_all.append(boxes_i)

        return imgs.to(device), boxes_all

    def forward(self, batch_items: List[Dict[str, Any]], num_lobes: int = 5, rationale_targets: Optional[List[str]] = None):
        device = next(self.parameters()).device
        imgs_5, boxes_all = self._build_lobe_tensor(batch_items, num_lobes=num_lobes, device=device)
        B = imgs_5.size(0)

        flat = imgs_5.view(B * num_lobes, 3, self.img_size, self.img_size)
        vec, feat_map = self.backbone(flat)
        vec = vec.view(B, num_lobes, 512)

        z = torch.tanh(self.feat_proj(vec))
        h0 = torch.zeros(1, B, z.size(-1), device=device)
        step_states, hn = self.gru(z, h0)
        final_h = hn.squeeze(0)

        logits = self.classifier(final_h)

        rationale = None
        rationale_loss = None
        if self.use_rationale:
            enc = self.to_t5(final_h).unsqueeze(1)
            if rationale_targets is not None:
                tok = self.tok(rationale_targets, padding=True, truncation=True, max_length=64, return_tensors="pt").to(device)
                labels = tok.input_ids.clone()
                labels[labels == self.tok.pad_token_id] = -100
                t5_out = self.t5(inputs_embeds=enc, labels=labels)
                rationale_loss = t5_out.loss
            else:
                gen_ids = self.t5.generate(inputs_embeds=enc, max_length=64, num_beams=3)
                rationale = [self.tok.decode(g, skip_special_tokens=True) for g in gen_ids]

        return {
            "logits": logits,
            "boxes": boxes_all,
            "feat_maps": feat_map,
            "flat_imgs": flat,
            "step_states": step_states,
            "rationale": rationale,
            "rationale_loss": rationale_loss
        }
