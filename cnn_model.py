"""DenseNet121 encoder: multi-label classifier (stage a) + feature extractor (stage b).

PROJECT_PLAN.md Section 3: pull from `densenet121.features(x)`, apply `F.relu` (torchvision's
own `forward()` does this before pooling, so the pretrained weights expect it), and then
*don't* global-average-pool - flatten the 7x7 grid into 49 spatial tokens of width 1024 so
the decoder can cross-attend over *where* a finding is.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import DenseNet121_Weights, densenet121

from labels import NUM_CONDITIONS

FEATURE_DIM = 1024   # DenseNet121 final feature width
GRID = 7             # 224 / 32 (DenseNet121 total stride)
NUM_REGIONS = GRID * GRID  # 49 spatial tokens


class DenseNet121Encoder(nn.Module):
    """Shared backbone. Exposes both a pooled vector (classification) and the 49 tokens."""

    def __init__(self, pretrained=True, num_classes=NUM_CONDITIONS, drop_rate=0.0):
        super().__init__()
        weights = DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = densenet121(weights=weights, drop_rate=drop_rate)
        self.features = backbone.features
        self.classifier = nn.Linear(FEATURE_DIM, num_classes)
        # CheXNet-style init: start every condition near its base rate rather than 0.5,
        # which stops the early steps from being dominated by the "everything negative"
        # prior on a dataset where most studies are normal.
        nn.init.zeros_(self.classifier.weight)
        nn.init.constant_(self.classifier.bias, -2.0)

    def forward_features(self, x):
        """(B, 3, 224, 224) -> (B, 49, 1024) spatial tokens."""
        f = self.features(x)          # (B, 1024, 7, 7)
        f = F.relu(f, inplace=True)   # matches torchvision's own forward()
        return f.flatten(2).transpose(1, 2).contiguous()  # (B, H*W, C)

    def forward_map(self, x):
        """(B, 3, 224, 224) -> (B, 1024, 7, 7), kept un-flattened for Grad-CAM."""
        return F.relu(self.features(x), inplace=True)

    def forward(self, x):
        """Multi-label logits (B, 14) - stage (a)."""
        f = self.forward_map(x)
        pooled = F.adaptive_avg_pool2d(f, 1).flatten(1)
        return self.classifier(pooled)

    # -- fine-tuning control (plan Section 3: frozen first, then denseblock4 + norm5) ----

    def freeze_backbone(self):
        for p in self.features.parameters():
            p.requires_grad = False
        return self

    def unfreeze_last_block(self):
        """Unfreeze only `denseblock4` + `norm5` - the joint fine-tune pass."""
        for name, module in self.features.named_children():
            if name in ("denseblock4", "norm5"):
                for p in module.parameters():
                    p.requires_grad = True
        return self

    def unfreeze_all(self):
        for p in self.features.parameters():
            p.requires_grad = True
        return self

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]


def load_encoder(checkpoint_path=None, pretrained=True, device="cpu", strict=False):
    """Build the encoder, optionally warm-starting from a stage (a) classifier checkpoint."""
    enc = DenseNet121Encoder(pretrained=pretrained)
    if checkpoint_path:
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
        missing, unexpected = enc.load_state_dict(state, strict=strict)
        if missing or unexpected:
            print(f"  encoder load: {len(missing)} missing, {len(unexpected)} unexpected keys")
    return enc.to(device)


@torch.no_grad()
def grad_cam_ready_map(encoder, images):
    """Convenience for week 7 Grad-CAM overlays: the raw (B, 1024, 7, 7) activation map."""
    encoder.eval()
    return encoder.forward_map(images)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    enc = DenseNet121Encoder(pretrained=False).to(device)
    x = torch.randn(2, 3, 224, 224, device=device)

    tokens = enc.forward_features(x)
    logits = enc(x)
    print(f"tokens {tuple(tokens.shape)}  (expect (2, {NUM_REGIONS}, {FEATURE_DIM}))")
    print(f"logits {tuple(logits.shape)}  (expect (2, {NUM_CONDITIONS}))")

    total = sum(p.numel() for p in enc.parameters())
    enc.freeze_backbone().unfreeze_last_block()
    trainable = sum(p.numel() for p in enc.trainable_parameters())
    print(f"params {total / 1e6:.2f}M, trainable after partial unfreeze "
          f"{trainable / 1e6:.2f}M ({100 * trainable / total:.1f}%)")
