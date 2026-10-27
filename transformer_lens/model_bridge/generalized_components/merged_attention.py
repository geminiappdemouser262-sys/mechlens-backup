"""Merged self/cross attention bridge (T5Gemma 2 decoder).

T5Gemma 2 replaces the decoder's self-attention + cross-attention pair with ONE attention
module (paper §2 "Merged Attention"):

    X : [b, m, d]  decoder residual (after pre-norm)        H : [b, n, d]  encoder output
    Q = q_norm(X W_q)                                       -> hook_q   [b, m,   n_heads,    d_head]
    K = k_norm([X ; H] W_k)                                 -> hook_k   [b, m+n, n_kv_heads, d_head]
    V =        [X ; H] W_v                                  -> hook_v   [b, m+n, n_kv_heads, d_head]
    S = Q K^T * scale + M          M: [m, m+n]              -> hook_attn_scores [b, n_heads, m, m+n]
    A = softmax(S)                                          -> hook_pattern     [b, n_heads, m, m+n]
    O = (A V) W_o                                           -> hook_z [b, m, n_heads, d_head], hook_out

There are NO separate cross-attention weights: the same W_k / W_v project both the decoder
slice and the encoder slice. This bridge therefore:

  * wraps the single HF attention module and *observes* it (masking, RoPE, KV-cache stay HF's);
  * exposes the standard AttentionBridge hooks with a key axis of length m + n;
  * adds read-only split views  hook_pattern_self  [..., :m]  and  hook_pattern_cross [..., m:]
    (and the same for attn_scores) so self-vs-cross analyses need no manual slicing;
  * records the decoder/encoder lengths of the last forward (``last_split``) for cache users.

Interventions should target ``hook_pattern`` / ``hook_k`` / ``hook_v`` (the real tensors);
the ``*_self`` / ``*_cross`` hooks receive views and return values are ignored.

CONFIRM-against-modeling_t5gemma2.py items are marked ``# CONFIRM``.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch

from transformer_lens.hook_points import HookPoint
from transformer_lens.model_bridge.generalized_components.attention import AttentionBridge

# Keyword under which HF passes the encoder output to the decoder attention module.
# CONFIRM: T5Gemma used `key_value_states` for cross-attn; Gemma-3-derived code may use
# `encoder_hidden_states`. Both are accepted; whichever is present is used.
_ENCODER_STATE_KWARGS: Tuple[str, ...] = ("encoder_hidden_states", "key_value_states")


class MergedAttentionBridge(AttentionBridge):
    """One attention module whose keys/values are [decoder ; encoder] (T5Gemma 2 decoder)."""

    hook_aliases = {
        **AttentionBridge.hook_aliases,
        # Legacy-style names some users reach for on encoder-decoder models.
        "hook_cross_pattern": "hook_pattern_cross",
        "hook_self_pattern": "hook_pattern_self",
    }

    def __init__(
        self,
        name: str,
        config: Any,
        submodules: Optional[Dict[str, Any]] = None,
        *,
        requires_position_embeddings: bool = True,  # Gemma-3-style RoPE on the decoder slice
        is_causal: bool = True,  # over the decoder slice; encoder slice is fully visible  # CONFIRM
        **kwargs: Any,
    ) -> None:
        # Never `is_cross_attention=True`: that flag makes AttentionBridge treat the module as a
        # second weight set with its own K/V, which double-counts W_k / W_v for merged attention.
        kwargs.pop("is_cross_attention", None)
        super().__init__(
            name,
            config,
            submodules,
            requires_position_embeddings=requires_position_embeddings,
            requires_attention_mask=True,
            is_causal=is_causal,
            is_cross_attention=False,
            **kwargs,
        )
        # Encoder states entering the shared K/V projections.
        self.hook_encoder_kv_in = HookPoint()
        # Read-only split views of the joint pattern / scores.
        self.hook_pattern_self = HookPoint()
        self.hook_pattern_cross = HookPoint()
        self.hook_attn_scores_self = HookPoint()
        self.hook_attn_scores_cross = HookPoint()
        self.is_merged_attention = True
        #: (decoder_len m, encoder_len n) of the most recent forward; None before any call.
        self.last_split: Optional[Tuple[int, int]] = None

    # ------------------------------------------------------------------ helpers ---------------
    @staticmethod
    def _pop_encoder_states(kwargs: Dict[str, Any]) -> Tuple[Optional[str], Optional[torch.Tensor]]:
        for key in _ENCODER_STATE_KWARGS:
            if kwargs.get(key) is not None:
                return key, kwargs[key]
        return None, None

    def _fire_split_views(self, joint: Optional[torch.Tensor], self_hook: HookPoint, cross_hook: HookPoint) -> None:
        """Fire the *_self / *_cross hooks on views of a [b, h, m, m+n] tensor. Return values ignored."""
        if joint is None or self.last_split is None or not isinstance(joint, torch.Tensor) or joint.dim() != 4:
            return
        m, _ = self.last_split
        if joint.shape[-1] < m:
            return
        self_hook(joint[..., :m])
        cross_hook(joint[..., m:])

    # ------------------------------------------------------------------ forward ---------------
    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """hook_in(X) + hook_encoder_kv_in(H) -> HF merged attention -> joint hooks -> split views.

        Delegation keeps HF's mask construction, RoPE application and EncoderDecoderCache
        behaviour authoritative. ``output_attentions`` must be on (adapter sets eager attention)
        for ``hook_pattern`` to receive the joint [b, h, m, m+n] weights.
        """
        if self.original_component is None:
            raise RuntimeError(f"Original component not set for {self.name}. Call set_original_component() first.")

        # Decoder input length m.
        x = kwargs.get("hidden_states", args[0] if args and isinstance(args[0], torch.Tensor) else None)
        m = int(x.shape[1]) if isinstance(x, torch.Tensor) and x.dim() >= 2 else 0

        # Encoder slice H: hook it, and record n. During cached decoding HF may pass H only on
        # the first step (encoder K/V then live in the cache); n is then read from the cache if
        # available, else from the last recorded split.  # CONFIRM cache layout
        enc_key, enc_states = self._pop_encoder_states(kwargs)
        if enc_states is not None:
            enc_states = self.hook_encoder_kv_in(enc_states)
            kwargs[enc_key] = enc_states  # type: ignore[index]
            n = int(enc_states.shape[1])
        elif self.last_split is not None:
            n = self.last_split[1]
        else:
            n = 0
        self.last_split = (m, n)

        output = super().forward(*args, **kwargs)  # fires hook_in, hook_out, hook_pattern, hook_attn_scores

        # Split views (no-ops when HF did not return attention weights).
        joint_pattern = output[1] if isinstance(output, tuple) and len(output) >= 2 else None
        self._fire_split_views(joint_pattern, self.hook_pattern_self, self.hook_pattern_cross)
        self._fire_split_views(joint_pattern, self.hook_attn_scores_self, self.hook_attn_scores_cross)
        # NOTE: AttentionBridge fires hook_attn_scores on the *post-softmax* weights when delegating
        # (HF returns no pre-softmax scores). For true pre-softmax scores use reference_forward()
        # below, which is what the parity tests compare against.  # CONFIRM if HF exposes scores
        return output

    # ------------------------------------------------------------------ reference math --------
    @torch.no_grad()
    def reference_forward(
        self,
        x: torch.Tensor,
        encoder_states: torch.Tensor,
        *,
        mask: torch.Tensor,
        rotate_q: Optional[Any] = None,
        rotate_k_decoder: Optional[Any] = None,
    ) -> Dict[str, torch.Tensor]:
        """Pure-PyTorch merged attention in the formula above, for tests and for pre-softmax scores.

        Args:
            x: decoder input [b, m, d] (post pre-norm).
            encoder_states: encoder output [b, n, d].
            mask: additive mask broadcastable to [b, n_heads, m, m+n] (0 / -inf).
            rotate_q / rotate_k_decoder: callables applying RoPE to Q and to the DECODER slice
                of K. RoPE treatment of the encoder slice is unspecified in the paper.  # CONFIRM

        Returns dict with q, k, v, scores, pattern, z, out (same layouts as the hooks).
        """
        cfg = self.config
        n_heads, d_head = cfg.n_heads, cfg.d_head
        n_kv = getattr(cfg, "n_key_value_heads", None) or n_heads
        b, m, _ = x.shape
        kv_in = torch.cat([x, encoder_states], dim=1)  # [b, m+n, d]

        q = self.q(x).view(b, m, n_heads, d_head)
        k = self.k(kv_in).view(b, -1, n_kv, d_head)
        v = self.v(kv_in).view(b, -1, n_kv, d_head)
        if hasattr(self.original_component, "q_norm"):  # Gemma-3 QK-norm, per head  # CONFIRM attr names
            q = self.original_component.q_norm(q)
            k = self.original_component.k_norm(k)  # applied to encoder slice too (same W_k)  # CONFIRM
        if rotate_q is not None:
            q = rotate_q(q)
        if rotate_k_decoder is not None:
            k = torch.cat([rotate_k_decoder(k[:, :m]), k[:, m:]], dim=1)

        if n_kv != n_heads:  # GQA: repeat kv heads
            rep = n_heads // n_kv
            k = k.repeat_interleave(rep, dim=2)
            v = v.repeat_interleave(rep, dim=2)
        scale = getattr(cfg, "attn_scale", None) or (d_head**0.5)  # Gemma-3: query_pre_attn_scalar**0.5
        scores = torch.einsum("bqhd,bkhd->bhqk", q, k) / scale + mask
        pattern = scores.softmax(-1)
        z = torch.einsum("bhqk,bkhd->bqhd", pattern, v)
        out = self.o(z.reshape(b, m, n_heads * d_head))
        return {"q": q, "k": k, "v": v, "scores": scores, "pattern": pattern, "z": z, "out": out}

    def get_random_inputs(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        """Component-test inputs: decoder states + encoder states (+ whatever the base needs)."""
        inputs = super().get_random_inputs(*args, **kwargs)
        hs = inputs.get("hidden_states")
        if isinstance(hs, torch.Tensor):
            b, m, d = hs.shape
            inputs[_ENCODER_STATE_KWARGS[0]] = torch.randn(b, max(4, m // 2), d, dtype=hs.dtype, device=hs.device)
        return inputs
