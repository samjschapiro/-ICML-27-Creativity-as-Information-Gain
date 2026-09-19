"""Teacher-forced conditional log-probabilities from a local MLX language model.

log p(target | context) is the sum of per-token log-probs of the target rendered as the assistant
turn after the context messages. Per-token values are attributed to labelled character spans of
the target so entity and relation tokens can be reported separately.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np


class Scorer:
    def __init__(self, model_name: str):
        from mlx_lm import load
        self.model_name = model_name
        self.model, self.tokenizer = load(model_name)
        hf = getattr(self.tokenizer, "_tokenizer", None)
        if hf is None or not getattr(hf, "is_fast", False):
            raise RuntimeError("FATAL: need a fast HF tokenizer for offset mapping")
        self._hf = hf

    def _prompt_ids(self, messages: list[dict]) -> list[int]:
        ids = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True)
        return list(ids)

    def _target_ids(self, target: str) -> tuple[list[int], list[tuple[int, int]]]:
        enc = self._hf(target, add_special_tokens=False, return_offsets_mapping=True)
        return list(enc["input_ids"]), list(enc["offset_mapping"])

    def score(self, messages: list[dict], target: str,
              spans: list[tuple[int, int, str]] | None = None) -> dict:
        """Returns total log-prob, token count, and per-kind sums over the labelled spans."""
        p_ids = self._prompt_ids(messages)
        t_ids, offsets = self._target_ids(target)
        if not t_ids:
            raise ValueError("FATAL: empty target")
        ids = mx.array(p_ids + t_ids)[None]
        logits = self.model(ids)
        # position i predicts token i+1; target tokens occupy positions len(p_ids) .. end
        pred = logits[0, len(p_ids) - 1: len(p_ids) - 1 + len(t_ids)].astype(mx.float32)
        logp = pred - mx.logsumexp(pred, axis=-1, keepdims=True)
        tok_lp = np.array(logp[mx.arange(len(t_ids)), mx.array(t_ids)])
        mx.eval(tok_lp) if isinstance(tok_lp, mx.array) else None
        out = {"logp": float(tok_lp.sum()), "n_tokens": len(t_ids), "token_logp": tok_lp.tolist()}
        if spans:
            by_kind: dict[str, float] = {}
            n_kind: dict[str, int] = {}
            for (a, b), lp in zip(offsets, tok_lp):
                kind = _kind_at(spans, a, b)
                by_kind[kind] = by_kind.get(kind, 0.0) + float(lp)
                n_kind[kind] = n_kind.get(kind, 0) + 1
            out["logp_by_kind"] = by_kind
            out["n_tokens_by_kind"] = n_kind
        return out


def _kind_at(spans: list[tuple[int, int, str]], a: int, b: int) -> str:
    """Label a token by the span holding most of it; separators count as 'sep'."""
    best, best_ov = "sep", 0
    for s, e, k in spans:
        ov = max(0, min(b, e) - max(a, s))
        if ov > best_ov:
            best, best_ov = k, ov
    return best
