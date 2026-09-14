"""Optional helpers for Hugging Face Llama-style (model.model.layers) and GPT-2/GPT-Neo-style (model.transformer.h) models: collect real residual states and build `map_fn`s that
apply a real sub-block (or a stretch of the network) to arbitrary states. The manifold code never imports this."""
from __future__ import annotations

import re

import torch

LOCATIONS_DOC = "'embed', 'L{k}.attn' (residual after the attention sub-layer of block k), 'L{k}.ffn' (block-k output)"


def _arch(model):
    """(embedding module, list of blocks, name of the pre-MLP norm, name of the MLP) for Llama-style and GPT-2/GPT-Neo-style models."""
    if hasattr(model, "model") and hasattr(model.model, "layers"):                      # Llama, Mistral, Qwen2, ...
        return model.model.embed_tokens, model.model.layers, "post_attention_layernorm", "mlp"
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):              # GPT-2, GPT-Neo, ...
        return model.transformer.wte, model.transformer.h, "ln_2", "mlp"
    raise ValueError("unsupported architecture: expected model.model.layers (Llama-style) or model.transformer.h (GPT-style)")


def locations(model):
    return ["embed"] + [f"L{k}.{s}" for k in range(len(_arch(model)[1])) for s in ("attn", "ffn")]


def _target(model, loc):
    embed, layers, norm2, _ = _arch(model)
    if loc == "embed":
        return embed, "out"
    k, part = re.fullmatch(r"L(\d+)\.(attn|ffn)", loc).groups()
    layer = layers[int(k)]
    return (getattr(layer, norm2), "pre") if part == "attn" else (layer, "out")


class _Hooks:
    """Capture residual states at `capture` locations; overwrite the state at `patch = {loc: (x, pos)}`."""

    def __init__(self, model, capture=(), patch=None):
        self.model, self.capture, self.patch, self.states, self.h = model, list(capture), patch or {}, {}, []

    def _touch(self, loc, h):
        if loc in self.patch:
            x, pos = self.patch[loc]
            h[torch.arange(h.shape[0], device=h.device), pos] = x.to(h.dtype)
        if loc in self.capture:
            self.states[loc] = h.detach().clone()
        return h

    def __enter__(self):
        for loc in dict.fromkeys(list(self.patch) + self.capture):
            mod, kind = _target(self.model, loc)
            if kind == "out":
                def hook(mod, args, out, loc=loc):
                    h = self._touch(loc, (out[0] if isinstance(out, tuple) else out).clone())
                    return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h
                self.h.append(mod.register_forward_hook(hook))
            else:                                  # in place: the residual add reuses this tensor
                def pre(mod, args, loc=loc):
                    self._touch(loc, args[0])
                self.h.append(mod.register_forward_pre_hook(pre))
        return self

    def __exit__(self, *a):
        for h in self.h:
            h.remove()


@torch.no_grad()
def collect_states(model, input_ids, positions, locs, attention_mask=None, batch_size=256):
    """Real residual states {loc: (N, D)} (preallocated on CPU) at `positions` (N,) of the sequences `input_ids` (N, L)."""
    dev = next(model.parameters()).device
    N, D = len(input_ids), _arch(model)[0].weight.shape[1]
    out = {l: torch.empty(N, D) for l in locs}
    for s in range(0, N, batch_size):
        ids, pos = input_ids[s:s + batch_size].to(dev), positions[s:s + batch_size].to(dev)
        mask = None if attention_mask is None else attention_mask[s:s + batch_size].to(dev)
        with _Hooks(model, locs) as hk:
            model(input_ids=ids, attention_mask=mask)
        for l in locs:
            out[l][s:s + len(ids)] = hk.states[l][torch.arange(len(ids), device=dev), pos].cpu()
    return out


def vocab_states(model, tok):
    """The embedding manifold: rows of the (tied) embedding matrix for all non-special tokens, and their ids."""
    embed = _arch(model)[0]
    ids = [v for v in range(embed.weight.shape[0]) if v not in set(tok.all_special_ids)]
    return embed.weight[ids].detach(), ids


def make_map(model, src, dst, context_ids, pos=None, batch_size=2048):
    """map_fn(x) = residual at `dst` when the residual at `src` (position `pos` of the fixed context) is replaced
    by x — the real Transformer stretch F_{src->dst} under a fixed surrounding context. For 'Lk.attn' -> 'Lk.ffn'
    this is the tokenwise FFN sub-layer and the context is unused."""
    dev = next(model.parameters()).device
    context_ids = torch.as_tensor(context_ids).to(dev)
    pos = context_ids.shape[-1] - 1 if pos is None else pos
    k = re.fullmatch(r"L(\d+)\.attn", src)
    if k and dst == f"L{k.group(1)}.ffn":
        _, layers, norm2, mlp = _arch(model)
        layer = layers[int(k.group(1))]

        @torch.no_grad()
        def ffn(x):
            x = x.to(dev).float()
            return torch.cat([xb + getattr(layer, mlp)(getattr(layer, norm2)(xb)) for xb in x.split(batch_size)])
        return ffn

    @torch.no_grad()
    def map_fn(x):
        x = x.to(dev).float(); out = []
        for xb in x.split(batch_size):
            ids = context_ids[None].expand(len(xb), -1)
            p = torch.full((len(xb),), pos, device=dev)
            with _Hooks(model, [dst], {src: (xb, p)}) as hk:
                model(input_ids=ids)
            out.append(hk.states[dst][torch.arange(len(xb), device=dev), p])
        return torch.cat(out)
    return map_fn
