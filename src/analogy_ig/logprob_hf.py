"""Teacher-forced conditional log-probabilities from a HuggingFace transformers model (CUDA).

Same contract as ``logprob.Scorer`` (mlx backend): ``score(messages, target, spans)`` returns the
total log-prob of the target rendered as the assistant turn, the token count, and per-kind sums.
Used on GPU boxes for the multi-model sensitivity check; the mlx backend stays for local runs.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


class ScorerHF:
    def __init__(self, model_name: str, dtype: str = "bfloat16"):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if not self.tokenizer.is_fast:
            raise RuntimeError("FATAL: need a fast tokenizer for offset mapping")
        if self.tokenizer.chat_template is None:
            raise RuntimeError(f"FATAL: {model_name} has no chat template; instruct models only")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=getattr(torch, dtype), device_map="cuda")
        self.model.eval()

    def _prompt_ids(self, messages: list[dict]) -> list[int]:
        # Some chat templates (gemma) reject a system role: fold it into the first user turn.
        try:
            ids = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True)
        except Exception as e:  # noqa: BLE001
            if messages[0]["role"] != "system":
                raise
            folded = [{"role": "user", "content": messages[0]["content"] + "\n\n" + messages[1]["content"]}]
            ids = self.tokenizer.apply_chat_template(folded, add_generation_prompt=True, tokenize=True)
        return list(ids)

    @torch.no_grad()
    def score(self, messages: list[dict], target: str,
              spans: list[tuple[int, int, str]] | None = None) -> dict:
        p_ids = self._prompt_ids(messages)
        enc = self.tokenizer(target, add_special_tokens=False, return_offsets_mapping=True)
        t_ids, offsets = list(enc["input_ids"]), list(enc["offset_mapping"])
        if not t_ids:
            raise ValueError("FATAL: empty target")
        ids = torch.tensor([p_ids + t_ids], device="cuda")
        logits = self.model(ids).logits[0]
        pred = logits[len(p_ids) - 1: len(p_ids) - 1 + len(t_ids)].float()
        logp = F.log_softmax(pred, dim=-1)
        tok_lp = logp[torch.arange(len(t_ids)), torch.tensor(t_ids, device="cuda")].cpu().numpy()
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


def _kind_at(spans, a, b) -> str:
    best, best_ov = "sep", 0
    for s, e, k in spans:
        ov = max(0, min(b, e) - max(a, s))
        if ov > best_ov:
            best, best_ov = k, ov
    return best
