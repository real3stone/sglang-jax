"""Fused EAGLE3 tree verify: device building blocks against the per-step tree path.

Each device function used inside the fused tree JIT has a host counterpart on
the per-step path. These tests feed both the same inputs and require the same
attention metadata, tree, KV movement and emitted tokens, all under an Explicit
mesh as in production.
"""

from functools import partial
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import AxisType, Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from sgl_jax.srt.kernels.speculative.build_eagle_tree_structure_kernel import QLEN_ONLY
from sgl_jax.srt.layers.attention.flashattention_backend import (
    FlashAttention,
    FlashAttentionMetadata,
    _draft_decode_kv_lens,
    draft_page_table_size,
    mask_row_width,
)
from sgl_jax.srt.mem_cache.memory_pool import MHATokenToKVPool
from sgl_jax.srt.model_executor.forward_batch_info import ForwardMode
from sgl_jax.srt.speculative.base_worker import BaseSpecWorker
from sgl_jax.srt.speculative.draft_extend_fused import (
    _make_eagle3_tree_draft_metadata,
    _make_eagle3_tree_verify_metadata,
)
from sgl_jax.srt.speculative.eagle_draft_worker import (
    select_top_k_tokens,
    update_eagle_lists,
)
from sgl_jax.srt.speculative.eagle_info import EagleDraftInput, EagleVerifyInput
from sgl_jax.srt.speculative.eagle_util import (
    accepted_path_kv_copies_device,
    build_tree_kernel_efficient,
    build_tree_kernel_efficient_device,
    copy_accepted_tree_kv,
    front_pack_accepted_tokens,
    front_pack_accepted_tokens_device,
)
from sgl_jax.srt.speculative.spec_info import SpeculativeAlgorithm


def _mesh():
    return Mesh(
        np.array(jax.devices()[:1]).reshape(1, 1),
        ("data", "tensor"),
        axis_types=(AxisType.Explicit, AxisType.Explicit),
    )


def _backend(mesh, page_size):
    return FlashAttention(
        num_attn_heads=2, num_kv_heads=1, head_dim=128, page_size=page_size, mesh=mesh
    )


def _aligned(lens, page_size):
    return -(-np.asarray(lens) // page_size) * page_size


def _cache_loc(alloc, page_size):
    """Token-level page table: request ``k``'s pages back to back, as padding_for_decode lays them."""
    chunks = []
    for k, length in enumerate(alloc):
        if length > 0:
            first_page = 16 * (k + 1)
            chunks.append(first_page * page_size + np.arange(_aligned(length, page_size)))
    cache_loc = np.concatenate(chunks).astype(np.int32)
    return np.pad(cache_loc, (0, max(0, 8192 - len(cache_loc))))


def _draft_lists(padded_bs, topk, steps, seed):
    """Run the real draft-step selection on random draft outputs.

    Returns the per-step parent arrays and the score / token / parents lists
    the tree builder consumes.
    """
    rng = np.random.default_rng(seed)
    topk_p = jnp.asarray(rng.random((padded_bs, topk), dtype=np.float32))
    topk_index = jnp.asarray(rng.integers(0, 50, (padded_bs, topk), dtype=np.int32))
    hidden = jnp.zeros((padded_bs, 2), dtype=jnp.float32)
    score_list = jnp.zeros((padded_bs, 1 + (steps - 1) * topk, topk), dtype=jnp.float32)
    token_list = jnp.zeros((padded_bs, topk + (steps - 1) * topk * topk), dtype=jnp.int32)
    parents_list = jnp.zeros((padded_bs, topk + 1 + (steps - 1) * topk), dtype=jnp.int32)
    scores = None
    parents_by_step = []
    for i in range(steps):
        _, hidden, scores, tree_info = select_top_k_tokens(
            i, topk_p, topk_index, hidden, scores, topk
        )
        score_list, token_list, parents_list = update_eagle_lists(
            i, score_list, token_list, parents_list, tree_info, topk
        )
        parents_by_step.append(tree_info[2])
        topk_p = jnp.asarray(rng.random((padded_bs * topk, topk), dtype=np.float32))
        topk_index = jnp.asarray(rng.integers(0, 50, (padded_bs * topk, topk), dtype=np.int32))
    return parents_by_step, (score_list, token_list, parents_list)


def _assert_same_layout(device, host, page_size):
    np.testing.assert_array_equal(np.asarray(device.cu_q_lens), np.asarray(host.cu_q_lens))
    np.testing.assert_array_equal(np.asarray(device.cu_kv_lens), np.asarray(host.cu_kv_lens))
    np.testing.assert_array_equal(np.asarray(device.seq_lens), np.asarray(host.seq_lens))
    np.testing.assert_array_equal(np.asarray(device.distribution), np.asarray(host.distribution))
    used = int(np.asarray(host.cu_kv_lens)[-1]) // page_size
    np.testing.assert_array_equal(
        np.asarray(device.page_indices)[:used], np.asarray(host.page_indices)[:used]
    )


def _assert_same_mask(device_mask, host_mask):
    """``device_mask`` may be wider than ``host_mask``; the extra columns stay masked."""
    device_mask, host_mask = np.asarray(device_mask), np.asarray(host_mask)
    width = host_mask.shape[-1]
    np.testing.assert_array_equal(device_mask[..., :width], host_mask)
    assert not device_mask[..., width:].any()


TREE_CASES = [
    ([5, 9, 3], 4, 2, 4),
    ([17, 6], 2, 3, 3),
    ([125, 4, 70], 4, 4, 4),  # the widest draft row crosses 128 within the round
    ([9000, 8000], 2, 2, 3),  # windows overflow the default draft page table at page_size 1
]


@pytest.mark.parametrize("extra_width", [0, 128])
@pytest.mark.parametrize("page_size", [1, 64])
@pytest.mark.parametrize("seq_lens, padded_bs, topk, steps", TREE_CASES)
def test_draft_step_metadata_matches_host(seq_lens, padded_bs, topk, steps, page_size, extra_width):
    mesh = _mesh()
    backend = _backend(mesh, page_size)
    real = len(seq_lens)
    seq = np.zeros(padded_bs, dtype=np.int32)
    seq[:real] = seq_lens
    alloc = np.where(seq > 0, seq + steps * topk, 0).astype(np.int32)
    batch = SimpleNamespace(
        cache_loc=_cache_loc(alloc, page_size),
        forward_mode=ForwardMode.DECODE,
        seq_lens=seq,
        logits_indices_selector=np.arange(real, dtype=np.int32),
        spec_info_padded=EagleDraftInput(allocate_lens=alloc),
        dp_size=1,
        per_dp_bs_size=padded_bs,
        speculative_num_steps=steps,
        speculative_eagle_topk=topk,
        spec_algorithm=SpeculativeAlgorithm.EAGLE3,
    )
    parents_by_step, _ = _draft_lists(padded_bs, topk, steps, seed=len(seq_lens))
    host_steps = backend.get_eagle_multi_step_metadata(batch)
    base = backend.get_eagle_base_metadata(batch)
    widest = _draft_decode_kv_lens(seq, max(steps - 2, 0), topk)
    width = mask_row_width(_aligned(widest, page_size)) + extra_width
    page_table_size = draft_page_table_size(
        seq[:real],
        np.zeros(real, dtype=np.int64),
        topk=topk,
        num_steps=steps,
        page_size=page_size,
        dp_size=1,
    )

    data = NamedSharding(mesh, P("data"))
    with jax.set_mesh(mesh):
        seq_device, alloc_device = jax.device_put((seq, alloc), data)
        for step in range(steps - 1):
            build = jax.jit(
                partial(
                    _make_eagle3_tree_draft_metadata,
                    step=step,
                    topk=topk,
                    width=width,
                    page_table_size=page_table_size,
                    page_size=page_size,
                    dp_size=1,
                )
            )
            device = build(base, seq_device, alloc_device, tuple(parents_by_step[1 : step + 1]))
            _assert_same_layout(device, host_steps[step], page_size)
            # The attention kernel derives its block sizes from the table length.
            np.testing.assert_array_equal(
                np.asarray(device.page_indices), np.asarray(host_steps[step].page_indices)
            )
            host_mask = backend.get_eagle_draft_decode_mask(batch, step, parents_by_step)
            _assert_same_mask(device.custom_mask, host_mask)
            assert device.custom_mask.sharding.is_equivalent_to(data, 3)


@pytest.mark.parametrize("extra_width", [0, 128])
@pytest.mark.parametrize("page_size", [1, 64])
@pytest.mark.parametrize(
    "verify_seq_lens, padded_bs, n, alloc_extra",
    [
        ([4, 11, 2], 4, 4, 6),
        ([130, 7], 2, 8, 16),  # per-round allocation longer than the verify window
    ],
)
def test_verify_metadata_matches_host(
    verify_seq_lens, padded_bs, n, alloc_extra, page_size, extra_width
):
    mesh = _mesh()
    backend = _backend(mesh, page_size)
    real = len(verify_seq_lens)
    verify = np.zeros(padded_bs, dtype=np.int32)
    verify[:real] = verify_seq_lens
    alloc = np.where(verify > 0, verify + 1 + alloc_extra, 0).astype(np.int32)
    tree_mask = np.random.default_rng(n).integers(0, 2, padded_bs * n * n).astype(np.int32)
    spec_info = EagleVerifyInput(
        draft_token=None,
        custom_mask=jnp.asarray(tree_mask),
        positions=None,
        retrive_index=None,
        retrive_next_token=None,
        retrive_next_sibling=None,
        spec_steps=n - 1,
        draft_token_num=n,
    )
    spec_info.allocate_lens = alloc[:real]
    batch = SimpleNamespace(
        cache_loc=_cache_loc(alloc, page_size),
        forward_mode=ForwardMode.TARGET_VERIFY,
        seq_lens=verify,
        logits_indices_selector=np.arange(real, dtype=np.int32),
        spec_info_padded=spec_info,
        dp_size=1,
        per_dp_bs_size=padded_bs,
    )
    host = backend.get_eagle_forward_metadata(batch)
    base = backend.get_eagle_base_metadata(batch)
    width = mask_row_width(_aligned(np.where(verify > 0, verify + n, 0), page_size)) + extra_width

    data = NamedSharding(mesh, P("data"))
    with jax.set_mesh(mesh):
        verify_device, alloc_device = jax.device_put((verify, alloc), data)
        build = jax.jit(
            partial(
                _make_eagle3_tree_verify_metadata,
                num_draft_tokens=n,
                width=width,
                page_size=page_size,
            )
        )
        device = build(
            FlashAttentionMetadata(page_indices=base.page_indices),
            verify_device,
            alloc_device,
            jax.device_put(tree_mask, NamedSharding(mesh, P())),
        )
    _assert_same_layout(device, host, page_size)
    _assert_same_mask(device.custom_mask, host.custom_mask)
    assert device.custom_mask.sharding.is_equivalent_to(data, 3)


@pytest.fixture
def pallas_interpret(monkeypatch):
    """Run Pallas kernels in interpret mode off TPU."""
    if jax.default_backend() != "tpu":
        import jax.experimental.pallas as pl
        from jax.experimental.pallas import tpu as pltpu

        pallas_call = pl.pallas_call
        interpret = pltpu.InterpretParams() if hasattr(pltpu, "InterpretParams") else True
        monkeypatch.setattr(pl, "pallas_call", partial(pallas_call, interpret=interpret))


@pytest.mark.parametrize("seq_lens, padded_bs, topk, steps", TREE_CASES)
def test_tree_build_in_jit_matches_host(seq_lens, padded_bs, topk, steps, pallas_interpret):
    mesh = _mesh()
    n = 1 + topk * 2
    real = len(seq_lens)
    seq = np.zeros(padded_bs, dtype=np.int32)
    seq[:real] = seq_lens
    verified_seq_lens = seq - 1
    verified_id = np.arange(100, 100 + padded_bs, dtype=np.int32)
    _, (scores, tokens, parents) = _draft_lists(padded_bs, topk, steps, seed=7)

    host = build_tree_kernel_efficient(
        verified_id,
        scores,
        tokens,
        parents,
        verified_seq_lens,
        np.sum(verified_seq_lens),
        topk,
        n,
        4096,
        padded_bs,
        steps,
        mesh,
        tree_mask_mode=QLEN_ONLY,
    )

    @jax.jit
    def build(verified_id, scores, tokens, parents, verified_seq_lens):
        return build_tree_kernel_efficient_device(
            verified_id,
            scores,
            tokens,
            parents,
            verified_seq_lens,
            jnp.sum(verified_seq_lens),
            topk,
            n,
            n,
            padded_bs,
            steps,
            tree_mask_mode=QLEN_ONLY,
        )

    replicated = NamedSharding(mesh, P())
    with jax.set_mesh(mesh):
        device = build(
            *jax.device_put((verified_id, scores, tokens, parents, verified_seq_lens), replicated)
        )
    names = [
        "tree_mask",
        "positions",
        "retrive_index",
        "retrive_next_token",
        "retrive_next_sibling",
    ]
    for name, d, h in zip(names + ["draft_tokens"], device, host):
        np.testing.assert_array_equal(np.asarray(d), np.asarray(h), err_msg=name)


N = 6  # verify draft tokens per request
WIDTH = 4  # accept window: speculative_num_steps + 1


def _kv_pool(page_size, size=2048, layer_num=2):
    """A KV pool whose every row holds ``slot + 1000 * layer``."""
    pool = MHATokenToKVPool(
        size=size,
        page_size=page_size,
        dtype=jnp.float32,
        head_num=1,
        head_dim=128,
        layer_num=layer_num,
        mesh=_mesh(),
    )
    for layer, kv in enumerate(pool.kv_buffer):
        values = np.arange(kv.shape[0] * kv.shape[1]).reshape(kv.shape[:2]) + 1000 * layer
        filled = np.broadcast_to(values[:, :, None, None, None], kv.shape).astype(np.float32)
        pool.kv_buffer[layer] = jax.device_put(filled, pool.kv_sharding)
    return pool


def _rows(pool, slots):
    """``(layer_num, len(slots))``: which slot's original KV each slot now holds."""
    out = []
    for layer, kv in enumerate(pool.kv_buffer):
        kv = np.asarray(kv)
        rows = kv.reshape(-1, *kv.shape[2:])[np.asarray(slots)]
        assert np.all(rows == rows[:, :1, :1, :1]), "a row mixes KV from several slots"
        out.append(rows[:, 0, 0, 0] - 1000 * layer)
    return np.stack(out).astype(np.int64)


def _verify_pool(page_size):
    """Two requests in pool rows 2 and 0, page-aligned, plus a padding slot.

    Returns the page table, the padding_for_decode cache_loc snapshot and the
    verify page table (pages repacked to each slot's KV window).
    """
    req_to_token = np.zeros((4, 64), dtype=np.int32)
    req_to_token[2] = 128 + np.arange(64)
    req_to_token[0] = 512 + np.arange(64)
    req_pool_indices = np.array([2, 0, 0], dtype=np.int32)
    window_starts = np.array([10, 20, 0], dtype=np.int32)
    cache_loc_starts = np.array([0, 40, -1])
    cache_loc = np.zeros(128, dtype=np.int32)
    cache_loc[0:40] = req_to_token[2, :40]
    cache_loc[40:80] = req_to_token[0, :40]
    kv_lens = np.where(np.arange(3) < 2, window_starts + N, 0)
    aligned = _aligned(kv_lens, page_size)
    pages = [
        req_to_token[req_pool_indices[s], : aligned[s] : page_size] // page_size for s in range(2)
    ]
    page_indices = np.zeros(64, dtype=np.int32)
    page_indices[: sum(len(p) for p in pages)] = np.concatenate(pages)
    cu_kv_lens = np.concatenate([[0], np.cumsum(aligned)]).astype(np.int32)
    return (
        req_to_token,
        req_pool_indices,
        cache_loc,
        cache_loc_starts,
        window_starts,
        page_indices,
        cu_kv_lens,
    )


def _accept_index(paths):
    """`(padded_bs, WIDTH)` flat node ids; padding slots get an empty path."""
    out = np.full((len(paths), WIDTH), -1, dtype=np.int32)
    for s, path in enumerate(paths):
        out[s, : len(path)] = s * N + np.asarray(path)
    return out


@pytest.mark.parametrize("page_size", [1, 8])
@pytest.mark.parametrize(
    "paths",
    [
        [[0, 2, 5], [0, 1, 3], []],
        [[0, 1, 2, 3], [0, 4], [0]],  # a chain, a sibling jump, a padding root
        [[0], [0, 1, 2, 5], []],
    ],
)
def test_accepted_path_kv_matches_host(page_size, paths):
    """Device KV copies leave each committed position with the KV the host path gives it.

    The host path compacts pointers at page_size 1 and copies KV when paged;
    the device path always copies KV and leaves the page table alone.
    """
    (
        req_to_token,
        req_pool_indices,
        cache_loc,
        cache_loc_starts,
        window_starts,
        page_indices,
        cu_kv_lens,
    ) = _verify_pool(page_size)
    before = req_to_token.copy()
    accept_index = _accept_index(paths)

    host_pool = _kv_pool(page_size)
    worker = SimpleNamespace(
        req_to_token_pool=SimpleNamespace(req_to_token=req_to_token),
        page_size=page_size,
        speculative_num_draft_tokens=N,
        target_worker=SimpleNamespace(model_runner=SimpleNamespace(token_to_kv_pool=host_pool)),
    )
    mwb = SimpleNamespace(
        req_pool_indices=req_pool_indices,
        cache_loc=cache_loc,
        draft_cache_loc_starts=cache_loc_starts,
        seq_lens=window_starts,
        logits_indices_selector=np.array([0, 1]),
    )
    BaseSpecWorker._move_accepted_paths_to_front(worker, mwb, accept_index)

    device_pool = _kv_pool(page_size)
    mesh = _mesh()
    with jax.set_mesh(mesh):
        src, dst = jax.jit(
            partial(accepted_path_kv_copies_device, draft_token_num=N, page_size=page_size)
        )(
            *jax.device_put(
                (accept_index, window_starts, page_indices, cu_kv_lens), NamedSharding(mesh, P())
            )
        )
    copy_accepted_tree_kv(
        device_pool,
        np.asarray(src),
        np.asarray(dst),
        num_pairs=accept_index.size - accept_index.shape[0],
    )

    for s, path in enumerate(paths[:2]):
        req = req_pool_indices[s]
        committed = window_starts[s] + len(path)
        np.testing.assert_array_equal(
            _rows(device_pool, before[req, :committed]),
            _rows(host_pool, req_to_token[req, :committed]),
        )


def test_accepted_tree_kv_copy_runs_only_for_moved_nodes():
    calls = []
    pool = SimpleNamespace(copy_kv_rows=lambda src, dst: calls.append((src.tolist(), dst.tolist())))
    src = np.array([7, 9, 4, 3], dtype=np.int32)
    copy_accepted_tree_kv(pool, src, np.full(4, -1, dtype=np.int32), num_pairs=3)
    assert calls == []
    copy_accepted_tree_kv(pool, src, np.array([-1, 12, -1, 13], dtype=np.int32), num_pairs=3)
    assert calls == [([9, 3, 0], [12, 13, 0])]


@pytest.mark.parametrize("draft_token_num, accept_width", [(4, 4), (8, 4), (3, 4)])
def test_front_pack_matches_host(draft_token_num, accept_width):
    bs = 3
    verified_id = np.random.default_rng(0).integers(1, 99, bs * accept_width).astype(np.int32)
    device = jax.jit(
        partial(
            front_pack_accepted_tokens_device,
            accept_width=accept_width,
            draft_token_num=draft_token_num,
        )
    )(jnp.asarray(verified_id))
    np.testing.assert_array_equal(
        np.asarray(device), front_pack_accepted_tokens(verified_id, accept_width, draft_token_num)
    )
