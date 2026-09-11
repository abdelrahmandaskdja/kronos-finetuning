import torch
import torch.nn as nn

from model import Kronos


class KronosDirectionRegressionModel(nn.Module):
    """Dual-head wrapper on top of Kronos for return regression + direction.

    Forward returns:
    - ret_pred: predicted horizon log-return (B,)
    - dir_logits: direction logits for BCEWithLogits (B,)
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

        stem_layers = []
        if use_ln:
            stem_layers.append(nn.LayerNorm(in_dim))

        if hidden > 0:
            stem_layers.append(nn.Linear(in_dim, hidden))
            stem_layers.append(nn.GELU())
            if dropout > 0.0:
                stem_layers.append(nn.Dropout(dropout))
            stem_layers.append(nn.Linear(hidden, hidden))
            stem_layers.append(nn.GELU())
            if dropout > 0.0:
                stem_layers.append(nn.Dropout(dropout))
            self.head_in_dim = hidden
        else:
            self.head_in_dim = in_dim

        self.shared_stem = nn.Sequential(*stem_layers) if stem_layers else nn.Identity()
        self.ret_head = nn.Linear(self.head_in_dim, 1)
        self.dir_head = nn.Linear(self.head_in_dim, 1)

    def forward(self, token_s1, token_s2, stamp):
        """
        token_s1: LongTensor (batch, seq_len)
        token_s2: LongTensor (batch, seq_len)
        stamp:    FloatTensor (batch, seq_len, time_dim)

        Returns:
            ret_pred: FloatTensor (batch,)
            dir_logits: FloatTensor (batch,)
        """
        s1_logits, context = self.kronos.decode_s1(
            s1_ids=token_s1,
            s2_ids=token_s2,
            stamp=stamp,
            padding_mask=None,
        )
        _ = s1_logits

        if self.pooling == "mean":
            feat = context.mean(dim=1)
        else:
            feat = context[:, -1, :]

        feat = self.shared_stem(feat)
        ret_pred = self.ret_head(feat).squeeze(-1)
        dir_logits = self.dir_head(feat).squeeze(-1)
        return ret_pred, dir_logits


def build_kronos_direction_regression_model(
    pretrained_predictor_path: str,
    head_hidden_dim: int = 512,
    head_dropout: float = 0.1,
    head_use_layernorm: bool = True,
    pooling: str = "last",
) -> KronosDirectionRegressionModel:
    """Build dual-head model from a pretrained Kronos predictor checkpoint."""
    kronos = Kronos.from_pretrained(pretrained_predictor_path)
    model = KronosDirectionRegressionModel(
        kronos=kronos,
        head_hidden_dim=head_hidden_dim,
        head_dropout=head_dropout,
        head_use_layernorm=head_use_layernorm,
        pooling=pooling,
    )
    return model
