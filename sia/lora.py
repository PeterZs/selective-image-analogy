"""LoRA layers of SIA.

``AnaLoRALinear`` is the always-active analogy LoRA (Stage 1). ``ExLoRALinear`` is the token-gated
exclusion LoRA (Stage 2): its update is applied only at the token positions set through
``set_token_mask`` (see ``sia.adapters.exlora_token_mask_ctx``).

The attribute names ``base``, ``lora_A`` and ``lora_B`` define the checkpoint keys and must not change.
"""
import math

import torch.nn as nn


class AnaLoRALinear(nn.Module):
    """Always-active LoRA wrapper around nn.Linear (no token mask)."""

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
        y = self.base(x)

        y_prev_dtype = y.dtype
        if y.dtype != self.lora_A.weight.dtype:
            x = x.to(self.lora_A.weight.dtype)
            y = y.to(self.lora_A.weight.dtype)

        lora = self.lora_B(self.lora_A(self.dropout(x))) * self.scaling
        return (y + lora).to(y_prev_dtype)


class ExLoRALinear(nn.Module):
    """LoRA wrapper whose update is applied only at masked token positions (no mask: base output only)."""

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

        # non-persistent buffer to hold the mask set via a context manager
        self.register_buffer("_token_mask", None, persistent=False)

    def set_scaling(self, scaling):
        self.scaling = scaling
    
    def reset_scaling(self):
        self.scaling = self.alpha / self.r

    def set_token_mask(self, mask): # mask: (B, S) bool
        self._token_mask = mask
    
    def forward(self, x):
        if self._token_mask is None:
            return self.base(x)

        y = self.base(x)

        y_prev_dtype = y.dtype
        if y.dtype != self.lora_A.weight.dtype:
            x = x.to(self.lora_A.weight.dtype)
            y = y.to(self.lora_A.weight.dtype)

        lora = self.lora_B(self.lora_A(self.dropout(x))) * self.scaling

        # x is (B, S, D); mask is (B, S)
        m = self._token_mask.to(lora.dtype).unsqueeze(-1)
        lora = lora * m

        return (y + lora).to(y_prev_dtype)

