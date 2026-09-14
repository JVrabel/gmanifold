"""Requires the SimpleStories-5M checkpoint (downloaded on first use)."""
import pytest
import torch

pytest.importorskip("transformers")
from gmanifold import transformer as tr


@pytest.fixture(scope="module")
def model_tok():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    name = "SimpleStories/SimpleStories-5M"
    model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32).cuda().eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, AutoTokenizer.from_pretrained(name)


@pytest.fixture(scope="module")
def cloud(model_tok):
    model, tok = model_tok
    prefix = tok("Once upon a time, there was a", add_special_tokens=False)["input_ids"]
    ids = torch.tensor([prefix + [v] for v in range(2, 402)])
    pos = torch.full((len(ids),), len(prefix))
    st = tr.collect_states(model, ids, pos, tr.locations(model))            # returned on CPU
    return ids, pos, {k: v.cuda() for k, v in st.items()}


def test_locations_and_vocab(model_tok):
    model, tok = model_tok
    assert tr.locations(model)[:3] == ["embed", "L0.attn", "L0.ffn"] and len(tr.locations(model)) == 13
    X, ids = tr.vocab_states(model, tok)
    assert X.shape == (4094, 256) and 2 in ids and 0 not in ids and 1 not in ids


def test_hooks_match_hidden_states(model_tok, cloud):
    model, _ = model_tok
    ids, pos, st = cloud
    with torch.no_grad():
        hs = model(ids[:64].cuda(), output_hidden_states=True).hidden_states
    r = torch.arange(64, device="cuda")
    assert torch.allclose(st["embed"][:64], hs[0][r, pos[:64].cuda()])
    assert torch.allclose(st["L2.ffn"][:64], hs[3][r, pos[:64].cuda()], atol=1e-6)
    layer = model.model.layers[1]
    with torch.no_grad():
        y = st["L1.attn"] + layer.mlp(layer.post_attention_layernorm(st["L1.attn"]))
    assert torch.allclose(y, st["L1.ffn"], atol=1e-5)


def test_make_map_reproduces_real_states(model_tok, cloud):
    model, _ = model_tok
    ids, pos, st = cloud
    ctx = ids[0]
    f = tr.make_map(model, "embed", "L3.ffn", ctx, pos=int(pos[0]))
    assert torch.allclose(f(st["embed"]), st["L3.ffn"], atol=1e-5)
    f2 = tr.make_map(model, "L1.attn", "L1.ffn", ctx)                     # tokenwise fast path
    assert torch.allclose(f2(st["L1.attn"]), st["L1.ffn"], atol=1e-5)
    f3 = tr.make_map(model, "L1.ffn", "L2.attn", ctx, pos=int(pos[0]))    # crosses attention
    assert torch.allclose(f3(st["L1.ffn"]), st["L2.attn"], atol=1e-5)
