import torch

from evident.losses import ebd_loss
from evident.model.eb_adapter import EBAdapter, EBContext

D, d, K, N, T, B = 64, 32, 4, 64, 3, 2


def _inputs():
    L = 10 + T * N + 12
    hs = torch.randn(B, L, D)
    vm = torch.zeros(B, L, dtype=torch.bool)
    vm[:, 10:10 + T * N] = True
    ctx = EBContext(
        subject_token_indices=[[L - 5], [L - 6, L - 5]],
        object_token_indices=[[L - 3], []],
        concept_valid_mask=torch.tensor([[True, True], [True, False]]),
    )
    return hs, vm, ctx


def test_zero_init_is_identity():
    adapter = EBAdapter(hidden_dim=D, bottleneck_dim=d, num_slots=K)
    hs, vm, ctx = _inputs()
    assert torch.count_nonzero(adapter(hs, vm, ctx)) == 0
    assert torch.count_nonzero(adapter.forward_text_only(hs)) == 0


def test_shapes_and_attention_normalization():
    adapter = EBAdapter(hidden_dim=D, bottleneck_dim=d, num_slots=K)
    hs, vm, ctx = _inputs()
    assert adapter(hs, vm, ctx).shape == hs.shape
    assert adapter.last_attn.shape == (B * T, K, N)
    torch.testing.assert_close(adapter.last_attn.sum(-1), torch.ones(B * T, K))  # Â sums to 1 over tokens
    assert adapter.last_gate.shape == (B, T)
    assert adapter.last_gate.min() >= 0 and adapter.last_gate.max() <= 1


def test_gate_ignores_missing_concepts():
    adapter = EBAdapter(hidden_dim=D, bottleneck_dim=d, num_slots=K)
    hs, vm, ctx = _inputs()
    adapter(hs, vm, ctx)
    gate_with_mask = adapter.last_gate[1].clone()
    # Sample 1 has no object: its gate is the normalized subject score alone.
    ctx_subject_only = EBContext(
        subject_token_indices=ctx.subject_token_indices,
        object_token_indices=[[], []],
        concept_valid_mask=torch.tensor([[True, False], [True, False]]),
    )
    adapter(hs, vm, ctx_subject_only)
    torch.testing.assert_close(adapter.last_gate[1], gate_with_mask)
    # Without a concept mask the gate is the identity.
    adapter(hs, vm, EBContext())
    assert torch.equal(adapter.last_gate, torch.ones(B, T))


def test_frames_are_processed_independently():
    adapter = EBAdapter(hidden_dim=D, bottleneck_dim=d, num_slots=K, use_e2v_gate=False)
    torch.nn.init.normal_(adapter.up_proj.weight)
    hs, vm, ctx = _inputs()
    out = adapter(hs, vm, ctx)
    hs2 = hs.clone()
    hs2[:, 10:10 + N] += 1.0  # perturb frame 0 only
    out2 = adapter(hs2, vm, ctx)
    changed = (out - out2).abs().amax(-1) > 0
    assert not changed[:, 10 + N:10 + T * N].any()


def test_ebd_loss_backpropagates_to_slot_attention():
    adapter = EBAdapter(hidden_dim=D, bottleneck_dim=d, num_slots=K)
    hs, vm, ctx = _inputs()
    adapter(hs, vm, ctx)
    loss = ebd_loss([adapter.last_attn], torch.randn(B * T, 256, 1024))
    loss.backward()
    assert loss.item() > 0
    assert adapter.slot_attn.to_q.weight.grad is not None
    assert adapter.down_proj.weight.grad is not None
