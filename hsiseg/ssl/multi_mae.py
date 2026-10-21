"""Multi-scene MAE: per-scene spectral stems + shared backbone.

Scenes come from different sensors with different band counts (IP 200, PaviaU 103,
Salinas 204), so only the 1x1 input projection and the reconstruction head are
scene-specific. Every Mamba/attention/decoder block is shared and therefore sees
all scenes' spatial-spectral structure.
"""
import torch.nn as nn
from hsiseg.models.dense_hmt import DenseHMT
from hsiseg.models.components import SpectralStem


class MultiSceneMAE(nn.Module):
    def __init__(self, bands_by_scene: dict, cfg):
        super().__init__()
        d = cfg.embed_dim
        # Backbone is built with in_bands=d, then its stem is removed: inputs
        # arrive already projected to D by the per-scene stem.
        self.backbone = DenseHMT(d, d, cfg)
        self.backbone.stem = nn.Identity()
        self.stems = nn.ModuleDict(
            {k: SpectralStem(b, d) for k, b in bands_by_scene.items()})
        self.recons = nn.ModuleDict({
            k: nn.Sequential(
                nn.Conv2d(d, d, 3, padding=1), nn.BatchNorm2d(d),
                nn.ReLU(inplace=True), nn.Conv2d(d, b, 1))
            for k, b in bands_by_scene.items()})

    def forward(self, x, scene: str):
        return self.recons[scene](self.backbone(self.stems[scene](x)))

    def encoder_state_dict(self, scene: str):
        """Keys match DenseHMT: this scene's stem.* + the shared stages/decoder."""
        sd = {f"stem.{k}": v for k, v in self.stems[scene].state_dict().items()}
        for k, v in self.backbone.state_dict().items():
            if k.startswith("head.") or k.startswith("stem."):
                continue
            sd[k] = v
        return sd
