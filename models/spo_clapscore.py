import torch
import torch.nn as nn

from .portable_m2d import PortableM2D


class SPO_CLAPScore(nn.Module):
    def __init__(self, cfg, device):
        super().__init__()
        self.device = device
        self.clap_encoder = PortableM2D(
            weight_file=cfg["clap_encoder"]["pretrained_model"],
            flat_features=True,
        ).to(self.device)

        # Freeze encoders
        if cfg["clap_encoder"]["freeze"]:
            self.clap_encoder.eval()
            for param in self.clap_encoder.parameters():
                param.requires_grad = False

        # Get standardize params
        self.standardize_layer = StandardizeLayer(
            mean=cfg["model"]["train_standard_stats"]["mean"],
            std=cfg["model"]["train_standard_stats"]["std"],
        )

    def forward(self, batch: dict, inference: bool = False):
        # text embedding: (B, 1024)
        text_embeds = self.clap_encoder.encode_clap_text(batch["caption_tokens"])
        text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
        # audio embedding: (B, 1024)
        audio_embeds = self.clap_encoder.encode_clap_audio(batch["wavs"])
        audio_embeds = audio_embeds / audio_embeds.norm(dim=-1, keepdim=True)

        # score predict
        logits_per_text = torch.matmul(text_embeds, audio_embeds.t())
        logits_per_audio = torch.matmul(audio_embeds, text_embeds.t())

        logits = (logits_per_text + logits_per_audio) / 2
        scores = torch.diag(logits)

        # クリッピング

        if inference:
            standard_score = scores * 10
        else:
            standard_score = self.standardize_layer(scores)
        return standard_score


class StandardizeLayer(nn.Module):
    """
    Standardization Layer
    Standardizes 0-1 scores obtained from cosine similarity.
    1. Rescale by a factor of 10.
    2. Standardize based on statistics from the training CSV.
    """

    def __init__(self, mean, std):
        super().__init__()
        self.mean = torch.tensor(mean)
        self.std = torch.tensor(std)

    def forward(self, x):
        x = x * 10
        return (x - self.mean.to(x.device)) / self.std.to(x.device)
