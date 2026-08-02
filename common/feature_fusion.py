"""
common/feature_fusion.py
=========================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------------------------------------------------------
# Strategy 1: Standard cosine similarity (CLIP default)
# -----------------------------------------------------------------------------

class CosineFusion(nn.Module):
    """
    Standard CLIP zero-shot: logit = scale * img_feat @ text_feat.T
    No learnable parameters -- pure cosine similarity.
    """
    def __init__(self, logit_scale_init=4.6052):
        super().__init__()
        self.logit_scale = nn.Parameter(
            torch.ones([]) * logit_scale_init, requires_grad=False
        )

    def forward(self, image_features, text_features):
        """
        image_features : (B, D) normalised
        text_features  : (n_cls, D) normalised
        Returns        : (B, n_cls) logits
        """
        scale = self.logit_scale.exp()
        return scale * image_features @ text_features.T


# -----------------------------------------------------------------------------
# Strategy 2: Concatenation fusion
# -----------------------------------------------------------------------------

class ConcatFusion(nn.Module):
    """
    Concatenation fusion:
      For each class c: concat [img_feat; text_feat_c] -> MLP -> scalar logit

    Architecture: Linear(2D -> D) -> GELU -> Linear(D -> 1)
    Produces one logit per class per image.

    Trainable parameters: ~2 * D^2 (small MLP)
    """
    def __init__(self, feat_dim=512, hidden_dim=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(feat_dim * 2, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, image_features, text_features):
        """
        image_features : (B, D)
        text_features  : (n_cls, D)
        Returns        : (B, n_cls)
        """
        B, D = image_features.shape
        n_cls = text_features.shape[0]

        # Expand for broadcasting: (B, n_cls, D)
        img_exp  = image_features.unsqueeze(1).expand(B, n_cls, D)
        text_exp = text_features.unsqueeze(0).expand(B, n_cls, D)

        # Concatenate: (B, n_cls, 2D)
        combined = torch.cat([img_exp, text_exp], dim=-1)

        # MLP: (B, n_cls, 1) -> (B, n_cls)
        logits = self.mlp(combined).squeeze(-1)
        return logits


# -----------------------------------------------------------------------------
# Strategy 3: Cross-attention fusion
# -----------------------------------------------------------------------------

class CrossAttentionFusion(nn.Module):
    """
    Cross-attention fusion:
      Image feature attends to all text class features.
      The attended representation is used to compute logits.

    Architecture:
      Q = Linear(img_feat)       -- query from image
      K = Linear(text_feat)      -- keys from all class text features
      V = Linear(text_feat)      -- values from all class text features
      attn_out = softmax(QK^T / sqrt(d)) * V
      logit_c  = dot(img_feat, attn_out_c)

    This allows the image to selectively attend to the most relevant
    class descriptions, rather than treating all classes equally.
    """
    def __init__(self, feat_dim=512, n_heads=8):
        super().__init__()
        self.feat_dim = feat_dim
        self.n_heads  = n_heads
        self.head_dim = feat_dim // n_heads
        assert feat_dim % n_heads == 0, "feat_dim must be divisible by n_heads"

        self.q_proj = nn.Linear(feat_dim, feat_dim, bias=False)
        self.k_proj = nn.Linear(feat_dim, feat_dim, bias=False)
        self.v_proj = nn.Linear(feat_dim, feat_dim, bias=False)
        self.out_proj = nn.Linear(feat_dim, feat_dim, bias=False)
        self.scale = self.head_dim ** -0.5

    def forward(self, image_features, text_features):
        """
        image_features : (B, D)
        text_features  : (n_cls, D)
        Returns        : (B, n_cls) logits
        """
        B, D = image_features.shape
        n_cls = text_features.shape[0]
        H = self.n_heads
        Hd = self.head_dim

        # Project
        Q = self.q_proj(image_features)          # (B, D)
        K = self.k_proj(text_features)            # (n_cls, D)
        V = self.v_proj(text_features)            # (n_cls, D)

        # Reshape for multi-head: (B, H, 1, Hd) and (1, H, n_cls, Hd)
        Q = Q.view(B, H, 1, Hd)
        K = K.view(1, H, n_cls, Hd).expand(B, -1, -1, -1)
        V = V.view(1, H, n_cls, Hd).expand(B, -1, -1, -1)

        # Attention weights: (B, H, 1, n_cls)
        attn = (Q @ K.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)

        # Attended values: (B, H, 1, Hd) -> (B, D)
        out = (attn @ V).squeeze(2)              # (B, H, Hd)
        out = out.reshape(B, D)
        out = self.out_proj(out)                 # (B, D)
        out = F.normalize(out, dim=-1)

        # Final logits: dot product of attended image repr with each text feat
        text_norm = F.normalize(text_features, dim=-1)   # (n_cls, D)
        logits = out @ text_norm.T                        # (B, n_cls)
        return logits * 100.0   # scale similar to CLIP


# -----------------------------------------------------------------------------
# Registry
# -----------------------------------------------------------------------------

FUSION_STRATEGIES = {
    "cosine":        CosineFusion,
    "concatenation": ConcatFusion,
    "cross_attention": CrossAttentionFusion,
}


def get_fusion(name, feat_dim=512, **kwargs):
    assert name in FUSION_STRATEGIES, \
        f"Unknown fusion: {name}. Choose from {list(FUSION_STRATEGIES.keys())}"
    return FUSION_STRATEGIES[name](feat_dim=feat_dim, **kwargs) \
        if name != "cosine" else FUSION_STRATEGIES[name]()
