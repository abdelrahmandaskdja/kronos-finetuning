import torch
import torch.nn as nn

from model import Kronos


class KronosDirectionModel(nn.Module):
    """Direction classification wrapper on top of a Kronos backbone.

    The output logits are binary-classification logits suitable for
    `BCEWithLogitsLoss`.

    If you optimize `model.parameters()`, both Kronos weights and this
    direction head are fine-tuned together.
    """

    def __init__(
        self,
        kronos: Kronos,
        head_hidden_dim: int = 512,
        head_dropout: float = 0.1,
        head_use_layernorm: bool = True,
        pooling: str = "last",
    ):
        super().__init__()
        self.kronos = kronos
        self.pooling = str(pooling).strip().lower()
        if self.pooling not in {"last", "mean"}:
            raise ValueError(f"Unsupported pooling={pooling}. Use 'last' or 'mean'.")

        in_dim = int(kronos.d_model)
        hidden = int(head_hidden_dim)
        dropout = float(max(0.0, head_dropout))
        use_ln = bool(head_use_layernorm)

        layers = []
        if use_ln:
            layers.append(nn.LayerNorm(in_dim))

        if hidden > 0:
            layers.append(nn.Linear(in_dim, hidden))
            layers.append(nn.GELU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            # Extra projection layer for a deeper direction head.
            layers.append(nn.Linear(hidden, hidden))
            layers.append(nn.GELU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            layers.append(nn.Linear(hidden, 1))
        else:
            layers.append(nn.Linear(in_dim, 1))

        self.direction_head = nn.Sequential(*layers)

    def forward(self, token_s1, token_s2, stamp):
        """
        token_s1: LongTensor (batch, seq_len)
        token_s2: LongTensor (batch, seq_len)
        stamp:    FloatTensor (batch, seq_len, time_dim)
        Returns:
            logits: FloatTensor (batch,) - binary logit per sample
        """
        # Use Kronos.decode_s1 to get context representation
        s1_logits, context = self.kronos.decode_s1(
            s1_ids=token_s1,
            s2_ids=token_s2,
            stamp=stamp,
            padding_mask=None,
        )
        _ = s1_logits
        # context: (B, T, d_model)
        if self.pooling == "mean":
            feat = context.mean(dim=1)  # (B, d_model)
        else:
            feat = context[:, -1, :]  # (B, d_model)
        logits = self.direction_head(feat)  # (B, 1)
        return logits.squeeze(-1)  # (B,)


def build_kronos_direction_model(
    pretrained_predictor_path: str,
    head_hidden_dim: int = 512,
    head_dropout: float = 0.1,
    head_use_layernorm: bool = True,
    pooling: str = "last",
) -> KronosDirectionModel:
    """Build direction classifier from a pretrained Kronos predictor checkpoint.

    Loads `Kronos.from_pretrained(pretrained_predictor_path)`, wraps it in
    `KronosDirectionModel`, and returns the wrapper.
    """
    kronos = Kronos.from_pretrained(pretrained_predictor_path)
    model = KronosDirectionModel(
        kronos=kronos,
        head_hidden_dim=head_hidden_dim,
        head_dropout=head_dropout,
        head_use_layernorm=head_use_layernorm,
        pooling=pooling,
    )
    return model
