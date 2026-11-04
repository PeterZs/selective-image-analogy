"""LoRA layers used by SIA.

``AnaLoRALinear`` is the always-active analogy LoRA (Stage 1). ``ExLoRALinear`` is the token-gated
exclusion LoRA (Stage 2): its update is applied only at the token positions set with
``set_token_mask`` (see ``sia.adapters.exlora_token_mask_ctx``).

The attribute names ``base``, ``lora_A`` and ``lora_B`` define the checkpoint keys
(``<module>.lora_A.weight`` etc.), so renaming them would break every released checkpoint.
"""
import math

import torch.nn as nn


class AnaLoRALinear(nn.Module):
    """Always-active LoRA around an ``nn.Linear``: ``y = base(x) + (alpha / r) * B(A(x))``.

    Args:
        base_linear: the frozen linear layer to wrap.
        r: LoRA rank.
        alpha: LoRA alpha; the update is scaled by ``alpha / r``.
        dropout: dropout probability on the LoRA input.

    ``lora_B`` is zero-initialized, so a freshly wrapped layer reproduces ``base_linear``.
    The LoRA branch runs in the dtype of the LoRA weights and the result is cast back to the
    dtype of the base output.
    """

    def __init__(self, base_linear: nn.Linear, r: int, alpha: int, dropout: float = 0.0):
        super().__init__()
        assert isinstance(base_linear, nn.Linear)

        self.base = base_linear
        self.in_features = base_linear.in_features
        self.out_features = base_linear.out_features
        self.alpha = alpha
        self.r = r
        self.scaling = alpha / r

        self.lora_A = nn.Linear(self.in_features, r, bias=False)
        self.lora_B = nn.Linear(r, self.out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        """Base output plus the scaled low-rank update, on every token."""
        y = self.base(x)

        y_prev_dtype = y.dtype
        if y.dtype != self.lora_A.weight.dtype:
            x = x.to(self.lora_A.weight.dtype)
            y = y.to(self.lora_A.weight.dtype)

        lora = self.lora_B(self.lora_A(self.dropout(x))) * self.scaling
        return (y + lora).to(y_prev_dtype)


class ExLoRALinear(nn.Module):
    """LoRA whose update is applied only at masked token positions.

    Args:
        base_module: the layer to wrap; any module with ``in_features`` / ``out_features``,
            typically an ``nn.Linear`` or an ``AnaLoRALinear`` (ExLoRA stacks on AnaLoRA).
        r: LoRA rank.
        alpha: LoRA alpha; the default scaling is ``alpha / r``.
        dropout: dropout probability on the LoRA input.

    With no mask set the layer returns ``base(x)`` exactly. With a ``(B, S)`` boolean mask the
    LoRA update is added only at the positions where the mask is true. The mask is a
    non-persistent buffer, so it never appears in a state dict.
    """

    def __init__(self, base_module, r: int, alpha: int, dropout: float = 0.0):
        super().__init__()
        assert hasattr(base_module, 'in_features') and hasattr(base_module, 'out_features'), \
            f"base_module must have in_features/out_features, got {type(base_module)}"

        self.base = base_module

        self.in_features = base_module.in_features
        self.out_features = base_module.out_features
        self.alpha = alpha
        self.r = r
        self.scaling = alpha / r

        self.lora_A = nn.Linear(self.in_features, r, bias=False)
        self.lora_B = nn.Linear(r, self.out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.register_buffer("_token_mask", None, persistent=False)

    def set_scaling(self, scaling):
        """Override the LoRA scaling factor (default ``alpha / r``)."""
        self.scaling = scaling

    def reset_scaling(self):
        """Restore the default scaling ``alpha / r``."""
        self.scaling = self.alpha / self.r

    def set_token_mask(self, mask):
        """Set the ``(B, S)`` boolean token mask, or ``None`` to disable the LoRA update."""
        self._token_mask = mask

    def forward(self, x):
        """Base output plus the scaled low-rank update on the masked tokens only (base output if no mask is set)."""
        if self._token_mask is None:
            return self.base(x)

        y = self.base(x)

        y_prev_dtype = y.dtype
        if y.dtype != self.lora_A.weight.dtype:
            x = x.to(self.lora_A.weight.dtype)
            y = y.to(self.lora_A.weight.dtype)

        lora = self.lora_B(self.lora_A(self.dropout(x))) * self.scaling

        # x is (B, S, D) and the mask is (B, S): broadcast over the feature dimension.
        m = self._token_mask.to(lora.dtype).unsqueeze(-1)
        lora = lora * m

        return (y + lora).to(y_prev_dtype)
