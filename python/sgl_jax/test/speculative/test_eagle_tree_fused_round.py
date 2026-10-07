"""Fused EAGLE3 tree decode against the per-step tree path, one round at a time.

Both paths run on one worker pair from the same state: page table, KV pools
and draft seeds. Each round must give the same accept lengths, emitted tokens,
next-round seeds, and KV at every committed position. The models are tiny,
with random weights and a four-token vocabulary so that drafts often match and
accepted paths vary. Off TPU, the attention kernel is replaced by a dense
reference that takes the same inputs and writes the same KV, and the other
Pallas kernels run in interpret mode.
"""

import json
from functools import partial
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import sgl_jax.srt.layers.attention.flashattention_backend as flashattention_backend
import sgl_jax.srt.speculative.draft_extend_fused as draft_extend_fused
import sgl_jax.srt.speculative.eagle_util as eagle_util
from sgl_jax.srt.kernels.ragged_paged_attention.ragged_paged_attention_v3 import (
    merge_kv,
)
from sgl_jax.srt.mem_cache.memory_pool import MHATokenToKVPool, _copy_kv_rows
from sgl_jax.srt.model_executor.forward_batch_info import CaptureHiddenMode, ForwardMode
from sgl_jax.srt.model_executor.model_runner import ModelRunner
from sgl_jax.srt.speculative.base_worker import BaseSpecWorker
from sgl_jax.srt.speculative.eagle_info import EagleDraftInput
from sgl_jax.srt.speculative.spec_info import SpeculativeAlgorithm

VOCAB = 4
MAX_KV = 512  # longest KV window the reference attention reads
RESERVE = 256  # KV positions owned by each request
NUM_REQS = 8


def reference_rpa(
    queries,
    keys,
    values,
    kv_cache,
    kv_lens,
    page_indices,
    cu_q_lens,
    cu_kv_lens,
    distribution,
    custom_mask,
    attention_sink,
    *,
    causal,
    sm_scale,
    sliding_window=None,
    soft_cap=None,
    xai_temperature_len=None,
    softmax_dtype=None,
    m_block_sizes=None,
):
    """Dense ragged paged attention over a single device's arguments."""
    assert sliding_window is None and soft_cap is None and xai_temperature_len is None
    assert attention_sink is None
    num_tokens, num_q_heads, head_dim = queries.shape
    num_kv_heads = keys.shape[1]
    num_pages, page_size = kv_cache.shape[:2]
    bs = kv_lens.shape[0]

    token = jnp.arange(num_tokens, dtype=jnp.int32)
    seq = jnp.searchsorted(cu_q_lens[1 : bs + 1], token, side="right").astype(jnp.int32)
    valid = seq < distribution[2]
    seq = jnp.minimum(seq, bs - 1)
    q_start = cu_q_lens[seq]
    q_len = cu_q_lens[seq + 1] - q_start
    kv_len = kv_lens[seq]
    q_pos = kv_len - q_len + (token - q_start)
    page_base = cu_kv_lens[seq] // page_size

    write_page = page_indices.at[page_base + q_pos // page_size].get(mode="fill", fill_value=0)
    write_page = jnp.where(valid, write_page, num_pages)
    new_rows = merge_kv(keys, values).astype(kv_cache.dtype)
    kv_cache = kv_cache.at[write_page, q_pos % page_size].set(new_rows, mode="drop")

    pos = jnp.arange(MAX_KV, dtype=jnp.int32)[None, :]
    read_page = page_indices.at[page_base[:, None] + pos // page_size].get(
        mode="fill", fill_value=0
    )
    slot = read_page * page_size + pos % page_size
    kv = kv_cache.reshape(num_pages * page_size, -1, kv_cache.shape[-1])[slot][..., :head_dim]
    k, v = kv[:, :, 0::2], kv[:, :, 1::2]

    allowed = (pos < kv_len[:, None]) & valid[:, None]
    if causal:
        allowed = allowed & (pos <= q_pos[:, None])
    else:
        width = custom_mask.shape[-1]
        keep = custom_mask[:, 0, :][:, jnp.minimum(pos[0], width - 1)]
        allowed = allowed & (keep != 0) & (pos < width)

    q = queries.reshape(num_tokens, num_kv_heads, num_q_heads // num_kv_heads, head_dim)
    scores = jnp.einsum("tngd,tknd->tngk", q.astype(jnp.float32), k.astype(jnp.float32))
    scores = jnp.where(allowed[:, None, None, :], scores * sm_scale, -1e30)
    probs = jnp.exp(scores - scores.max(axis=-1, keepdims=True))
    probs = jnp.where(allowed[:, None, None, :], probs, 0.0)
    denom = probs.sum(axis=-1, keepdims=True)
    probs = probs / jnp.where(denom > 0, denom, 1.0)
    out = jnp.einsum("tngk,tknd->tngd", probs, v.astype(jnp.float32))
    return out.reshape(num_tokens, num_q_heads, head_dim).astype(queries.dtype), kv_cache


@pytest.fixture(scope="module")
def off_tpu_kernels():
    if jax.default_backend() == "tpu":
        yield
        return
    import jax.experimental.pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    choose_backend = ModelRunner._get_attention_backend

    def flash_attention_backend(self):
        # The model runner picks the native backend whenever the device is CPU.
        device = self.server_args.device
        self.server_args.device = "tpu"
        try:
            return choose_backend(self)
        finally:
            self.server_args.device = device

    interpret = pltpu.InterpretParams() if hasattr(pltpu, "InterpretParams") else True
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(flashattention_backend, "ragged_paged_attention_v3", reference_rpa)
        patch.setattr(ModelRunner, "_get_attention_backend", flash_attention_backend)
        patch.setattr(pl, "pallas_call", partial(pl.pallas_call, interpret=interpret))
        yield


def _write_configs(root):
    common = {
        "hidden_size": 128,
        "intermediate_size": 256,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 128,
        "vocab_size": VOCAB,
        "max_position_embeddings": 2048,
        "rms_norm_eps": 1e-6,
        "rope_theta": 10000.0,
        "rope_parameters": {"rope_type": "default", "rope_theta": 10000.0},
        "tie_word_embeddings": False,
        "torch_dtype": "float32",
        "hidden_act": "silu",
    }
    target, draft = root / "target", root / "draft"
    target.mkdir()
    draft.mkdir()
    (target / "config.json").write_text(
        json.dumps(
            {
                **common,
                "architectures": ["Qwen3ForCausalLM"],
                "model_type": "qwen3",
                "num_hidden_layers": 4,
                "attention_bias": False,
                "eagle_config": {"eagle_aux_hidden_state_layer_ids": [0, 1, 2]},
            }
        )
    )
    (draft / "config.json").write_text(
        json.dumps(
            {
                **common,
                "architectures": ["LlamaForCausalLMEagle3"],
                "model_type": "llama",
                "num_hidden_layers": 1,
                "draft_vocab_size": VOCAB,
            }
        )
    )
    return str(target), str(draft)


def _randomize(runner, seed, int_leaf=None):
    """Replace a runner's float weights with random values; dummy loading leaves zeros."""
    rng = np.random.default_rng(seed)
    for i, leaf in enumerate(runner.model_state_leaves):
        if not jnp.issubdtype(leaf.dtype, jnp.floating):
            if int_leaf is not None and leaf.shape == int_leaf.shape:
                runner.model_state_leaves[i] = jax.device_put(
                    int_leaf.astype(leaf.dtype), leaf.sharding
                )
            continue
        if leaf.ndim == 1:
            values = 1.0 + 0.1 * rng.standard_normal(leaf.shape)
        else:
            values = rng.standard_normal(leaf.shape) / np.sqrt(leaf.shape[-1])
        runner.model_state_leaves[i] = jax.device_put(values.astype(leaf.dtype), leaf.sharding)


class _Harness:
    """One target + EAGLE3 worker pair and the decode state both paths start from."""

    def __init__(self, root, page_size, topk, steps, num_draft_tokens, tp_size=1, dtype="float32"):
        from sgl_jax.srt.managers.tp_worker import ModelWorker
        from sgl_jax.srt.server_args import ServerArgs
        from sgl_jax.srt.speculative.eagle_worker import EAGLEWorker
        from sgl_jax.srt.utils.mesh_utils import create_device_mesh

        target_path, draft_path = _write_configs(root)
        args = ServerArgs(
            model_path=target_path,
            speculative_draft_model_path=draft_path,
            speculative_algorithm="EAGLE3",
            speculative_eagle_topk=topk,
            speculative_num_steps=steps,
            speculative_num_draft_tokens=num_draft_tokens,
            page_size=page_size,
            tp_size=tp_size,
            attention_backend="fa",
            disable_overlap_schedule=True,
            load_format="dummy",
            dtype=dtype,
            skip_tokenizer_init=True,
            disable_radix_cache=True,
            random_seed=0,
            mem_fraction_static=0.1,
            max_total_tokens=8192,
            max_running_requests=NUM_REQS,
            context_length=1024,
            disable_precompile=True,
            precompile_bs_paddings=[4, 8],
            precompile_token_paddings=[64],
            chunked_prefill_size=64,
            max_prefill_tokens=64,
        )
        mesh = create_device_mesh(ici_parallelism=[1, tp_size], dcn_parallelism=[1, 1])
        self.target_worker = ModelWorker(server_args=args, mesh=mesh)
        self.spec_worker = EAGLEWorker(server_args=args, target_worker=self.target_worker)
        self.page_size = page_size
        self.num_draft_tokens = num_draft_tokens
        draft_worker = self.spec_worker.draft_worker
        hot_token_ids = np.random.default_rng(9).permutation(VOCAB).astype(np.int32)
        _randomize(self.target_worker.model_runner, 3)
        _randomize(draft_worker.draft_model_runner, 4, int_leaf=hot_token_ids)
        draft_worker.hot_token_ids = jax.device_put(
            hot_token_ids, draft_worker.hot_token_ids.sharding
        )

        self.target_pool = self.target_worker.model_runner.token_to_kv_pool
        self.draft_pool = draft_worker.draft_model_runner.token_to_kv_pool
        self.req_to_token = self.target_worker.model_runner.req_to_token_pool.req_to_token
        rng = np.random.default_rng(page_size)
        num_pages = self.target_pool.kv_buffer[0].shape[0]
        pages_per_req = RESERVE // page_size
        free_pages = rng.permutation(np.arange(1, num_pages))[: NUM_REQS * pages_per_req]
        for req in range(NUM_REQS):
            pages = free_pages[req * pages_per_req : (req + 1) * pages_per_req]
            self.req_to_token[req, :RESERVE] = (
                pages[:, None] * page_size + np.arange(page_size)
            ).reshape(-1)
        for pool, seed in ((self.target_pool, 1), (self.draft_pool, 2)):
            fill = np.random.default_rng(seed)
            for layer, kv in enumerate(pool.kv_buffer):
                values = fill.standard_normal(kv.shape).astype(kv.dtype)
                pool.kv_buffer[layer] = jax.device_put(values, pool.kv_sharding)

    def seeds(self, seq_lens, seed):
        """Draft seeds of freshly prefilled requests, as the scheduler holds them."""
        n = len(seq_lens)
        rng = np.random.default_rng(seed)
        topk = self.spec_worker.topk
        hidden = self.spec_worker.draft_worker.model_config.hidden_size
        return EagleDraftInput(
            topk_p=np.sort(rng.random((n, topk), dtype=np.float32), axis=1)[:, ::-1].copy(),
            topk_index=np.stack([rng.permutation(VOCAB)[:topk] for _ in range(n)]).astype(np.int32),
            hidden_states=rng.standard_normal((n, hidden)).astype(np.float32),
            verified_id=rng.integers(0, VOCAB, n).astype(np.int32),
            allocate_lens=np.asarray(seq_lens, dtype=np.int32),
        )

    def snapshot(self):
        return (
            [np.asarray(kv) for kv in self.target_pool.kv_buffer],
            [np.asarray(kv) for kv in self.draft_pool.kv_buffer],
            self.req_to_token.copy(),
        )

    def restore(self, snapshot):
        target_kv, draft_kv, req_to_token = snapshot
        for pool, buffers in ((self.target_pool, target_kv), (self.draft_pool, draft_kv)):
            for layer, kv in enumerate(buffers):
                pool.kv_buffer[layer] = jax.device_put(kv, pool.kv_sharding)
        self.req_to_token[...] = req_to_token

    def run(self, fused, reqs, seq_lens, seeds, snapshot, alloc_extra=0):
        """One decode round from ``snapshot``; returns its outputs and the state after it.

        Each request has ``alloc_extra`` more KV slots than the scheduler would give it.
        """
        self.restore(snapshot)
        spec_worker = self.spec_worker
        spec_worker._can_use_fused_eagle3_tree = fused
        real_bs = len(reqs)
        total_bs = 4 if real_bs <= 4 else 8
        pad = total_bs - real_bs
        mwb = spec_worker.draft_worker.compilation_manager._make_dummy_batch(
            total_bs,
            total_bs,
            ForwardMode.DECODE,
            spec_worker.precompile_cache_loc_paddings[-1],
            speculative_algorithm=SpeculativeAlgorithm.EAGLE3,
            dp_size=1,
            per_dp_bs_size=total_bs,
        )
        mwb.seq_lens = np.pad(seq_lens, (0, pad)).astype(np.int32)
        mwb.req_pool_indices = np.pad(reqs, (0, pad)).astype(np.int32)
        mwb.real_bs = real_bs
        mwb.real_bs_per_dp = [real_bs]
        mwb.logits_indices_selector = np.arange(real_bs, dtype=np.int32)
        allocate_lens = seq_lens + EagleDraftInput.ALLOC_LEN_PER_DECODE - 1 + alloc_extra
        mwb.spec_info_padded = EagleDraftInput(
            topk_p=np.pad(seeds.topk_p, ((0, pad), (0, 0))),
            topk_index=np.pad(seeds.topk_index, ((0, pad), (0, 0))),
            hidden_states=np.pad(seeds.hidden_states, ((0, pad), (0, 0))),
            verified_id=np.pad(seeds.verified_id, (0, pad)),
            capture_hidden_mode=CaptureHiddenMode.FULL,
            allocate_lens=np.pad(allocate_lens, (0, pad)).astype(np.int32),
        )
        mwb.capture_hidden_mode = CaptureHiddenMode.FULL

        out = spec_worker.forward_batch_speculative_generation(mwb)
        accept = np.asarray(out.accept_lens)[:real_bs]
        emitted = np.asarray(out.next_token_ids).reshape(total_bs, self.num_draft_tokens)
        draft = out.next_draft_input
        return {
            "padded_bs": total_bs,
            "accept": accept,
            "emitted": [emitted[i, : accept[i]] for i in range(real_bs)],
            "seeds": EagleDraftInput(
                topk_p=np.asarray(draft.topk_p),
                topk_index=np.asarray(draft.topk_index),
                hidden_states=np.asarray(draft.hidden_states),
                verified_id=np.asarray(draft.verified_id),
                allocate_lens=np.asarray(draft.allocate_lens),
            ),
            "snapshot": self.snapshot(),
        }

    def committed_kv(self, result, reqs, seq_lens):
        """KV behind every committed position of each request, target then draft.

        Target verify starts at ``seq_lens - 1``; the draft extend writes one
        position earlier.
        """
        target_kv, draft_kv, req_to_token = result["snapshot"]
        out = []
        for kvs, end in (
            (target_kv, seq_lens - 1 + result["accept"]),
            (draft_kv, seq_lens - 2 + result["accept"]),
        ):
            for req, length in zip(reqs, end):
                slots = req_to_token[req, :length]
                out.append(np.stack([kv.reshape(-1, *kv.shape[2:])[slots] for kv in kvs]))
        return out


def _assert_same_round(legacy, fused, harness, reqs, seq_lens):
    np.testing.assert_array_equal(fused["accept"], legacy["accept"])
    for got, want in zip(fused["emitted"], legacy["emitted"]):
        np.testing.assert_array_equal(got, want)
    for name in ("topk_index", "verified_id"):
        np.testing.assert_array_equal(
            getattr(fused["seeds"], name), getattr(legacy["seeds"], name), err_msg=name
        )
    for name in ("topk_p", "hidden_states"):
        np.testing.assert_allclose(
            getattr(fused["seeds"], name),
            getattr(legacy["seeds"], name),
            rtol=1e-6,
            atol=1e-6,
            err_msg=name,
        )
    for got, want in zip(
        harness.committed_kv(fused, reqs, seq_lens), harness.committed_kv(legacy, reqs, seq_lens)
    ):
        np.testing.assert_allclose(got, want, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize(
    "page_size, topk, steps, num_draft_tokens, tp_size",
    [
        (16, 2, 3, 4, 1),
        (1, 2, 3, 4, 1),
        (64, 4, 5, 8, 1),
        pytest.param(
            16,
            2,
            3,
            4,
            2,
            marks=pytest.mark.skipif(len(jax.devices()) < 2, reason="needs 2 devices"),
        ),
    ],
)
def test_fused_rounds_match_per_step_path(
    tmp_path, off_tpu_kernels, monkeypatch, page_size, topk, steps, num_draft_tokens, tp_size
):
    harness = _Harness(tmp_path, page_size, topk, steps, num_draft_tokens, tp_size)
    assert harness.spec_worker._can_use_fused_eagle3_tree

    moves = []
    move_paths = BaseSpecWorker._move_accepted_paths_to_front

    def record_moves(self, model_worker_batch, accept_index):
        nodes = accept_index - np.arange(len(accept_index))[:, None] * num_draft_tokens
        moves.append(int(np.sum((accept_index >= 0) & (nodes != np.arange(steps + 1)))))
        return move_paths(self, model_worker_batch, accept_index)

    monkeypatch.setattr(BaseSpecWorker, "_move_accepted_paths_to_front", record_moves)
    copies = []
    copy_kv_rows = MHATokenToKVPool.copy_kv_rows

    def record_copy(self, src, dst):
        copies.append(len(src))
        return copy_kv_rows(self, src, dst)

    monkeypatch.setattr(MHATokenToKVPool, "copy_kv_rows", record_copy)
    launches = []

    def recorded(name):
        launch = getattr(draft_extend_fused, name)

        def record(*args, **kwargs):
            launches.append(name)
            return launch(*args, **kwargs)

        return record

    for name in ("launch_eagle3_tree_verify", "launch_eagle3_tree_draft_extend"):
        monkeypatch.setattr(draft_extend_fused, name, recorded(name))
    # The draft lists the tree is built from carry every draft step's scores,
    # so they expose draft-step errors that leave the chosen tree unchanged.
    trees = []
    preprocess = eagle_util.build_tree_kernel_efficient_preprocess

    def record_tree(verified_id, scores, tokens, parents, *args, **kwargs):
        jax.debug.callback(
            lambda *lists: trees.append([np.asarray(x) for x in lists]),
            verified_id,
            scores,
            tokens,
            parents,
        )
        return preprocess(verified_id, scores, tokens, parents, *args, **kwargs)

    monkeypatch.setattr(eagle_util, "build_tree_kernel_efficient_preprocess", record_tree)

    reqs = np.array([5, 2, 6])
    # 126 puts the widest window one column past 128, the first mask-width bucket.
    seq_lens = np.array([37, 126, 9])
    seeds = harness.seeds(seq_lens, seed=0)
    snapshot = harness.snapshot()
    for round_id in range(4):
        if round_id == 2:
            # Request 2 finishes; requests 0 and 7 join with fresh prefill seeds.
            keep = np.array([0, 2])
            seeds.filter_batch(keep, has_been_filtered=False)
            joined = np.array([21, 50])
            seeds.merge_batch(harness.seeds(joined, seed=round_id))
            reqs = np.concatenate([reqs[keep], [0, 7]])
            seq_lens = np.concatenate([seq_lens[keep], joined])
        legacy = harness.run(False, reqs, seq_lens, seeds, snapshot)
        copies.clear()
        fused = harness.run(True, reqs, seq_lens, seeds, snapshot)
        # One copy padded to the batch size, and none when no node moved.
        assert copies == ([fused["padded_bs"] * steps] if moves[-1] else [])
        jax.effects_barrier()
        legacy_tree, fused_tree = trees[-2:]
        for name, got, want in zip(
            ("verified_id", "scores", "tokens", "parents"), fused_tree, legacy_tree
        ):
            np.testing.assert_allclose(got, want, rtol=1e-6, atol=1e-6, err_msg=name)
        _assert_same_round(legacy, fused, harness, reqs, seq_lens)
        seq_lens = seq_lens + legacy["accept"]
        seeds, snapshot = legacy["seeds"], legacy["snapshot"]

    assert launches == ["launch_eagle3_tree_verify", "launch_eagle3_tree_draft_extend"] * 4
    assert len(trees) == 8
    assert sum(moves) > 0, "no round moved an accepted path; the KV copy went untested"


@pytest.mark.parametrize(
    "changes, expected",
    [
        ({}, True),
        ({"is_all_greedy": False}, False),
        ({"return_logprob": True}, False),
        ({"return_output_logprob_only": True}, False),
        ({"can_use": False}, False),
    ],
)
def test_rounds_outside_the_fused_path_take_the_per_step_path(changes, expected):
    worker = SimpleNamespace(_can_use_fused_eagle3_tree=changes.get("can_use", True))
    batch = SimpleNamespace(
        sampling_info=SimpleNamespace(is_all_greedy=changes.get("is_all_greedy", True)),
        return_logprob=changes.get("return_logprob", False),
        return_output_logprob_only=changes.get("return_output_logprob_only", False),
    )
    assert BaseSpecWorker._use_fused_eagle3_tree(worker, batch) is expected


def test_precompile_covers_the_fused_tree_jits(tmp_path, off_tpu_kernels, monkeypatch):
    """Rounds after precompile hit the compiled JITs, with bf16 draft seeds."""
    harness = _Harness(tmp_path, 16, 2, 3, 4, dtype="bfloat16")
    spec_worker = harness.spec_worker
    spec_worker.precompile_spec_decode()
    draft_worker = spec_worker.draft_worker
    jits = (
        draft_worker._fused_eagle3_tree_verify_jit_fn,
        draft_worker._fused_eagle3_tree_draft_extend_jit_fn,
        _copy_kv_rows,
    )
    compiled = [jit._cache_size() for jit in jits]
    copies = []
    copy_kv_rows = MHATokenToKVPool.copy_kv_rows

    def record_copy(self, src, dst):
        copies.append(len(src))
        return copy_kv_rows(self, src, dst)

    monkeypatch.setattr(MHATokenToKVPool, "copy_kv_rows", record_copy)

    reqs, seq_lens = np.array([5, 2, 6]), np.array([37, 64, 9])
    seeds = harness.seeds(seq_lens, seed=0)
    seeds.topk_p = seeds.topk_p.astype(jnp.bfloat16)
    seeds.hidden_states = seeds.hidden_states.astype(jnp.bfloat16)
    snapshot = harness.snapshot()
    for _ in range(2):
        result = harness.run(True, reqs, seq_lens, seeds, snapshot)
        seq_lens = seq_lens + result["accept"]
        seeds, snapshot = result["seeds"], result["snapshot"]
    assert copies, "no round moved a node; the KV copy's precompile went unchecked"
    assert [jit._cache_size() for jit in jits] == compiled


def test_tree_windows_must_fit_the_allocation(tmp_path, off_tpu_kernels):
    # 3/2/4 needs seq_lens + 3 slots: the verify window and the last draft step.
    harness = _Harness(tmp_path, 1, 2, 3, 4)
    reqs, seq_lens = np.array([5, 2]), np.array([37, 9])
    with pytest.raises(AssertionError, match="tree windows need"):
        harness.run(
            True,
            reqs,
            seq_lens,
            harness.seeds(seq_lens, seed=0),
            harness.snapshot(),
            alloc_extra=-3,
        )
