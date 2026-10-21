"""MAE wrapper: DenseHMT encoder + light reconstruction head.

Pretraining reconstructs the original cube from a corrupted one, so no labels
are used. Afterwards the encoder weights transfer to the segmentation model.
"""
import torch
import torch.nn as nn

from hsiseg.models.dense_hmt import DenseHMT


class DenseHMT_MAE(nn.Module):
    def __init__(self, in_bands, cfg):
        super().__init__()
        # reuse the full DenseHMT but replace the classification head
        self.backbone = DenseHMT(in_bands, cfg.embed_dim, cfg)   # out channels = D
        self.recon = nn.Sequential(
            nn.Conv2d(cfg.embed_dim, cfg.embed_dim, 3, padding=1),
            nn.BatchNorm2d(cfg.embed_dim), nn.ReLU(inplace=True),
            nn.Conv2d(cfg.embed_dim, in_bands, 1))

    def forward(self, x_corrupt):
        return self.recon(self.backbone(x_corrupt))

    def encoder_state_dict(self):
        """Weights to transfer: everything except the final head."""
        return {k: v for k, v in self.backbone.state_dict().items()
                if not k.startswith("head.")}
