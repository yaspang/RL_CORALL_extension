import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from env import OWN_DIM, INTR_DIM, GOAL_DIM
from cases import MAX_INTRUDERS

OBS_DIM = OWN_DIM + MAX_INTRUDERS * INTR_DIM + GOAL_DIM


class TargetAttentionExtractor(BaseFeaturesExtractor):
    """
    Permutation-invariant feature extractor for multi-target collision avoidance.

    Observation layout: [own(OWN_DIM) | intr_0..N(MAX_INTRUDERS*INTR_DIM) | goal(GOAL_DIM)]
    Intruder features: [x_rel, y_rel, vx_rel, vy_rel, sin_dpsi, cos_dpsi, dr] in own body frame.

    Own vessel + goal are encoded into a query vector.
    Each intruder is encoded with a shared linear layer into key/value vectors.
    Cross-attention (own→intruders) aggregates intruder context.
    Padding slots (all-zero rows) are masked out.
    """

    def __init__(
        self,
        observation_space,
        embed_dim: int = 64,
        num_heads: int = 4,
        features_dim: int = 256,
    ):
        super().__init__(observation_space, features_dim)

        self.own_encoder = nn.Sequential(
            nn.Linear(OWN_DIM + GOAL_DIM, embed_dim),
            nn.ReLU(),
        )

        self.intruder_encoder = nn.Sequential(
            nn.Linear(INTR_DIM, embed_dim),
            nn.ReLU(),
        )

        self.attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

        self.final = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, features_dim),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        own  = obs[:, :OWN_DIM]
        tgts = obs[:, OWN_DIM : OWN_DIM + MAX_INTRUDERS * INTR_DIM]
        goal = obs[:, OWN_DIM + MAX_INTRUDERS * INTR_DIM:]

        tgts = tgts.view(-1, MAX_INTRUDERS, INTR_DIM)

        pad_mask = (tgts.abs().sum(dim=-1) == 0.0)
        all_masked = pad_mask.all(dim=-1, keepdim=True)
        pad_mask = pad_mask & ~all_masked.expand_as(pad_mask)

        own_emb = self.own_encoder(torch.cat([own, goal], dim=-1))  # [B, E]
        tgt_emb = self.intruder_encoder(tgts)                       # [B, N, E]

        query = own_emb.unsqueeze(1)
        attn_out, _ = self.attention(
            query, tgt_emb, tgt_emb, key_padding_mask=pad_mask
        )
        attn_out = attn_out.squeeze(1)                              # [B, E]

        return self.final(torch.cat([own_emb, attn_out], dim=-1))   # [B, features_dim]


class MLPExtractor(BaseFeaturesExtractor):
    """
    Flat MLP feature extractor. Concatenates the full observation and passes
    it through a stack of Linear→ReLU layers.
    """

    def __init__(
        self,
        observation_space,
        hidden_sizes: list[int] = None,
        features_dim: int = 256,
    ):
        super().__init__(observation_space, features_dim)
        if hidden_sizes is None:
            hidden_sizes = [256, 256]

        layers: list[nn.Module] = []
        in_dim = OBS_DIM
        for h in hidden_sizes:
            layers += [nn.Linear(in_dim, h), nn.ReLU()]
            in_dim = h
        layers.append(nn.Linear(in_dim, features_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        own  = obs[:, :OWN_DIM]
        tgts = obs[:, OWN_DIM : OWN_DIM + MAX_INTRUDERS * INTR_DIM].view(-1, MAX_INTRUDERS, INTR_DIM)
        goal = obs[:, OWN_DIM + MAX_INTRUDERS * INTR_DIM:]

        pad_mask = (tgts.abs().sum(dim=-1) == 0.0)                    # [B, N]
        tgts = tgts * (~pad_mask).unsqueeze(-1)                       # zero padded slots
        tgts = tgts.view(-1, MAX_INTRUDERS * INTR_DIM)

        return self.net(torch.cat([own, tgts, goal], dim=-1))
