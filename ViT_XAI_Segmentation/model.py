"""
ViT-Seg + Concept-Guided XAI model.

Architecture adaptation of Wickramanayake et al., arXiv:2101.03919
("Comprehensible CNNs via Guided Concept Learning") to:
  1) Vision Transformer encoder (not CNN) — per advisor feedback
  2) Dense binary forest segmentation (not image classification)

Pipeline:
  Image
    → Patch embed + ViT encoder (attention = built-in XAI)
    → Concept layer (1x1 over token features → K concept maps)
    → Segmentation head (upsample concepts → forest mask)
    → Concept contributions (GAP × classifier weights) for explanations
"""
from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import config as cfg


# ── ViT building blocks ──────────────────────────────────────────────────────

class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_ch=3, embed_dim=192):
        super().__init__()
        self.grid = img_size // patch_size
        self.num_patches = self.grid * self.grid
        self.proj = nn.Conv2d(in_ch, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # B,C,H,W → B,N,D
        x = self.proj(x)  # B,D,Gh,Gw
        B, D, Gh, Gw = x.shape
        x = x.flatten(2).transpose(1, 2)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=3, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.last_attn: Optional[torch.Tensor] = None  # for XAI

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        self.last_attn = attn.detach()  # B, heads, N, N
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj_drop(self.proj(x))
        return x


class MLP(nn.Module):
    def __init__(self, dim, mlp_ratio=4.0, drop=0.0):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.drop(self.act(self.fc1(x)))
        x = self.drop(self.fc2(x))
        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, drop=0.0, attn_drop=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads, attn_drop, drop)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, mlp_ratio, drop)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class VisionTransformerEncoder(nn.Module):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        embed_dim=192,
        depth=6,
        num_heads=3,
        mlp_ratio=4.0,
        drop=0.1,
        attn_drop=0.0,
    ):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, 3, embed_dim)
        n = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, n + 1, embed_dim))
        self.pos_drop = nn.Dropout(drop)
        self.blocks = nn.ModuleList(
            [Block(embed_dim, num_heads, mlp_ratio, drop, attn_drop) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.grid = img_size // patch_size
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
          tokens: B, N, D  (patch tokens only, no CLS)
          cls:    B, D
        """
        x = self.patch_embed(x)
        B, N, D = x.shape
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = self.pos_drop(x + self.pos_embed)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x[:, 1:, :], x[:, 0]


# ── Full model ───────────────────────────────────────────────────────────────

class ViTConceptSeg(nn.Module):
    """
    ViT encoder + concept layer (paper) + segmentation head.

    Explanations:
      - attention maps from last ViT block
      - concept activation maps
      - concept contribution scores to forest decision (paper §III-C)
    """

    def __init__(self):
        super().__init__()
        self.encoder = VisionTransformerEncoder(
            img_size=cfg.IMG_SIZE,
            patch_size=cfg.PATCH_SIZE,
            embed_dim=cfg.EMBED_DIM,
            depth=cfg.DEPTH,
            num_heads=cfg.NUM_HEADS,
            mlp_ratio=cfg.MLP_RATIO,
            drop=cfg.DROP_RATE,
            attn_drop=cfg.ATTN_DROP,
        )
        D = cfg.EMBED_DIM
        K = cfg.NUM_CONCEPTS
        G = self.encoder.grid

        # Concept layer: 1x1 conv over token feature map (paper uses 1x1 on last conv)
        self.concept_proj = nn.Conv2d(D, K, kernel_size=1, bias=True)

        # Segmentation from concept maps (+ residual from tokens)
        self.seg_fuse = nn.Sequential(
            nn.Conv2d(K + D, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 1),
        )

        # Paper: GAP → FC for class decision / contributions
        # Binary forest: 2-way (forest / background) contribution readout
        self.concept_fc = nn.Linear(K, 2)

        # Joint embedding for mapping consistency (paper L_s)
        self.visual_embed = nn.Linear(G * G, cfg.CONCEPT_EMBED_DIM)  # flatten each concept map
        self.phrase_embed = nn.Embedding(K, cfg.CONCEPT_EMBED_DIM)

        self.concept_names = list(cfg.CONCEPT_NAMES)

    def tokens_to_map(self, tokens: torch.Tensor) -> torch.Tensor:
        B, N, D = tokens.shape
        G = int(math.sqrt(N))
        return tokens.transpose(1, 2).reshape(B, D, G, G)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        tokens, cls = self.encoder(x)
        feat = self.tokens_to_map(tokens)          # B,D,G,G
        concepts = self.concept_proj(feat)         # B,K,G,G

        # GAP concept activations (paper \hat v_k)
        gap = concepts.mean(dim=(2, 3))            # B,K
        gap_centered = gap - gap.mean(dim=1, keepdim=True)

        # Segmentation logits at patch resolution → upsample to image
        fused = torch.cat([concepts, feat], dim=1)
        seg_low = self.seg_fuse(fused)             # B,1,G,G
        seg_logits = F.interpolate(
            seg_low, size=(cfg.IMG_SIZE, cfg.IMG_SIZE),
            mode="bilinear", align_corners=False,
        )

        # Concept → class logits (for contribution explanations)
        class_logits = self.concept_fc(gap)        # B,2

        return {
            "seg_logits": seg_logits,
            "concepts": concepts,
            "gap": gap,
            "gap_centered": gap_centered,
            "class_logits": class_logits,
            "tokens": tokens,
        }

    @torch.no_grad()
    def concept_contributions(self, gap: torch.Tensor) -> torch.Tensor:
        """
        Paper §III-C: contribution of concept k to class c = gap_k * W[k,c]
        Returns softmax percentages over concepts for the forest class (index 1).
        Shape: B,K
        """
        W = self.concept_fc.weight  # 2,K
        # forest class row
        contrib = gap * W[1].unsqueeze(0)  # B,K
        # shift to positive for softmax display
        contrib = contrib - contrib.min(dim=1, keepdim=True).values
        return F.softmax(contrib, dim=1)

    def attention_rollout(self) -> Optional[torch.Tensor]:
        """Average last-block attention over heads, CLS→patches (B, G, G)."""
        attn = self.encoder.blocks[-1].attn.last_attn
        if attn is None:
            return None
        # B, heads, N+1, N+1  (with CLS)
        a = attn.mean(dim=1)  # B, N+1, N+1
        cls_to_patch = a[:, 0, 1:]  # B, N
        G = self.encoder.grid
        return cls_to_patch.reshape(-1, 1, G, G)


# ── Losses (paper-adapted) ───────────────────────────────────────────────────

def dice_loss_with_logits(logits, targets, eps=1.0):
    probs = torch.sigmoid(logits)
    probs = probs.view(probs.size(0), -1)
    targets = targets.view(targets.size(0), -1)
    inter = (probs * targets).sum(dim=1)
    dice = (2 * inter + eps) / (probs.sum(dim=1) + targets.sum(dim=1) + eps)
    return (1 - dice).mean()


def segmentation_loss(logits, targets):
    bce = F.binary_cross_entropy_with_logits(logits, targets)
    dice = dice_loss_with_logits(logits, targets)
    return cfg.BCE_WEIGHT * bce + cfg.DICE_WEIGHT * dice


def concept_uniqueness_loss(gap_centered: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    """Paper L_u: BCE between sigmoid(normalized GAP) and concept indicators."""
    y = torch.sigmoid(gap_centered)
    return F.binary_cross_entropy(y, z)


def mapping_consistency_loss(
    concepts: torch.Tensor,
    z: torch.Tensor,
    visual_embed: nn.Linear,
    phrase_embed: nn.Embedding,
    concepts_counter: torch.Tensor,
    z_counter: torch.Tensor,
) -> torch.Tensor:
    """
    Paper L_m = L_s + L_c
    concepts: B,K,G,G ; z: B,K
    """
    B, K, G, _ = concepts.shape
    flat = concepts.view(B, K, G * G)
    # embed each concept map
    f = visual_embed(flat)  # B,K,E
    f = F.normalize(f, dim=-1)
    u = F.normalize(phrase_embed.weight, dim=-1)  # K,E

    # Semantic triplet L_s
    # for each present concept k, f_k closer to u_k than u_k'
    sim = torch.einsum("bke,je->bkj", f, u)  # B,K,K
    eye = torch.arange(K, device=concepts.device)
    pos = sim[:, eye, eye]  # B,K
    # max over negatives
    neg_mask = ~torch.eye(K, dtype=torch.bool, device=concepts.device)
    neg = sim.masked_fill(~neg_mask.unsqueeze(0), -1e9).max(dim=-1).values  # B,K
    Ls = F.relu(neg - pos + cfg.ALPHA_TRIPLET)
    Ls = (Ls * z).sum() / (z.sum() + 1e-6)

    # Counter loss L_c
    flat_c = concepts_counter.view(B, K, G * G)
    f_c = F.normalize(visual_embed(flat_c), dim=-1)
    sim_pos = (f * u.unsqueeze(0)).sum(dim=-1)     # B,K
    sim_neg = (f_c * u.unsqueeze(0)).sum(dim=-1)   # B,K
    present_diff = (z - z_counter).clamp(min=0)    # concepts in i not in i'
    Lc = F.relu(sim_neg - sim_pos + cfg.BETA_COUNTER)
    Lc = (Lc * present_diff).sum() / (present_diff.sum() + 1e-6)

    return Ls + Lc


def total_loss(outputs, targets, z, outputs_counter, z_counter, model) -> Dict[str, torch.Tensor]:
    """
    Paper: L = L_A + λ (L_u + L_m)
    Here L_A = segmentation BCE+Dice (+ optional concept-class CE).
    """
    L_seg = segmentation_loss(outputs["seg_logits"], targets)
    # auxiliary: concept GAP should also predict forest/background from mask mean
    forest_label = (targets.view(targets.size(0), -1).mean(dim=1) > 0.5).long()
    L_cls = F.cross_entropy(outputs["class_logits"], forest_label)
    L_A = L_seg + 0.2 * L_cls

    L_u = concept_uniqueness_loss(outputs["gap_centered"], z)
    L_m = mapping_consistency_loss(
        outputs["concepts"],
        z,
        model.visual_embed,
        model.phrase_embed,
        outputs_counter["concepts"],
        z_counter,
    )
    L = L_A + cfg.LAMBDA_CONCEPT * (L_u + L_m)
    return {"loss": L, "L_A": L_A.detach(), "L_u": L_u.detach(), "L_m": L_m.detach(), "L_seg": L_seg.detach()}
