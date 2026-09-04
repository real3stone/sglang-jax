"""Exp 3 — the cost side of "is tree speculative decoding worth it on this backend?"

The decision is one inequality:

    tau_tree / tau_chain  >  T_tree / T_chain

The left side is an accept-rate question (Exp 1/2, needs a real model pair on
v6e-4). This script measures the right side, which needs no model at all — only
kernels — so it runs on a single v6e chip, and it can settle the question on its
own: if the cost ratio is higher than any plausible accept-rate gain, stop.

WHAT IS MEASURED
  A. target-verify attention as a function of draft_token_num, chain vs tree.
     Chain (topk=1) passes no mask and runs causal=1; tree (topk>1) passes the
     rank-3 tree mask and runs causal=0. That asymmetry is production behaviour,
     not an experimental choice: flashattention_backend forces causal=0 exactly
     when a custom mask is present, and eagle_draft_worker sets tree_mask=None
     exactly when topk==1.
  B. the three speculative Pallas kernels (tree construction, tree-greedy
     verify, tree sampling) at the fixtures the repo's own benchmark uses.
     Chain drafting runs none of them.

WHAT IS NOT MEASURED, AND WHICH WAY IT BIASES
  The draft model forward. Tree drafting runs it on bs*topk rows per step
  instead of bs*1, and that cost is dominated by the model's MLP, which needs
  weights. Leaving it out UNDERSTATES the tree's cost, so the ratio printed here
  is a LOWER BOUND. A tree that already fails the break-even on this number
  fails for real.

  Also excluded: host-side tree bookkeeping (build_tree_mask_for_draft_decode is
  still a Python triple loop), and the loss of overlap scheduling, which topk>1
  forces off (server_args.py, the overlap gate). Both also work against the tree.

USAGE (Colab v6e-1)
    !cd /content/sglang-jax && python /path/to/exp3_tree_cost.py
    # or:  python exp3_tree_cost.py --bs 8 --kv-len 4096
"""

from __future__ import annotations

import argparse
import functools
import statistics
import sys
import types

# Run from the repo root. On a bare clone sgl_jax lives under python/ and the
# benchmark package sits at the top level, so put both on the path rather than
# requiring a pip install (reinstalling jax on a hosted TPU runtime is risky).
sys.path.insert(0, "python")
sys.path.insert(0, ".")

import jax
import jax.numpy as jnp
import numpy as np

REPO_HINT = "run this from the root of an sglang-jax checkout"


# --------------------------------------------------------------------------
# environment
# --------------------------------------------------------------------------
def check_env() -> None:
    print("=" * 72)
    print("environment")
    print("=" * 72)
    try:
        import jaxlib
    except ImportError:  # pragma: no cover
        sys.exit("jaxlib missing; " + REPO_HINT)
    devices = jax.devices()
    print(f"  jax {jax.__version__} / jaxlib {jaxlib.__version__}")
    print(f"  devices: {devices}")
    if devices[0].platform != "tpu":
        sys.exit(
            "No TPU. This measures TPU kernel time; a CPU number is meaningless.\n"
            "On Colab: Runtime > Change runtime type > TPU."
        )
    try:
        import libtpu

        print(f"  libtpu {getattr(libtpu, '__version__', 'unknown')}")
    except ImportError:
        print("  libtpu: not importable as a module (normal on some images)")

    # Pallas smoke test. Some hosted images pin a libtpu that cannot lower
    # Mosaic at all, and every measurement below would fail in a confusing way.
    from jax.experimental import pallas as pl

    def _kernel(x_ref, o_ref):
        o_ref[...] = x_ref[...] + 1

    try:
        out = pl.pallas_call(
            _kernel,
            out_shape=jax.ShapeDtypeStruct((8, 128), jnp.float32),
        )(jnp.zeros((8, 128), jnp.float32))
        assert float(out[0, 0]) == 1.0
        print("  pallas: OK")
    except Exception as exc:  # pragma: no cover
        sys.exit(f"Pallas smoke test failed, nothing below will work:\n{exc}")
    print()


# --------------------------------------------------------------------------
# A. target-verify attention
# --------------------------------------------------------------------------
def bench_verify(
    *,
    batch_size: int,
    draft_token_num: int,
    kv_len: int,
    with_mask: bool,
    q_head_num: int,
    kv_head_num: int,
    head_dim: int,
    page_size: int,
    tries: int,
) -> float:
    """Median ms for one target-verify attention call."""
    from benchmark.kernels.flash_attention.utils import (
        create_target_verify_uniform_data,
        create_tree_mask_rank3,
    )
    from sgl_jax.srt.kernels.ragged_paged_attention.ragged_paged_attention_v3 import (
        get_vmem_limit,
        ragged_paged_attention,
    )
    from sgl_jax.srt.kernels.utils.perf import multiple_iteration_timeit_from_trace

    prefix_len = kv_len - draft_token_num
    if prefix_len <= 0:
        raise ValueError(f"kv_len {kv_len} must exceed draft_token_num {draft_token_num}")
    pages_per_seq = -(-kv_len // page_size)
    # Returns a 12-tuple, not a dict:
    #   q, k, v, kv_cache, kv_lens, page_indices, cu_q_lens, cu_kv_lens,
    #   num_seqs, seq_lens, cache_loc, distribution
    (q, k, v, kv_cache, kv_lens, page_indices, cu_q_lens, cu_kv_lens, _num_seqs, _seq_lens,
     _cache_loc, distribution) = create_target_verify_uniform_data(
        batch_size=batch_size,
        draft_token_num=draft_token_num,
        prefix_len=prefix_len,
        page_indices_capacity=batch_size * pages_per_seq + 1,
        max_kv_cache_tokens=batch_size * pages_per_seq * page_size,
        q_head_num=q_head_num,
        kv_head_num=kv_head_num,
        head_dim=head_dim,
        page_size=page_size,
    )
    mask = (
        create_tree_mask_rank3(
            batch_size=batch_size, draft_token_num=draft_token_num, kv_len=kv_len
        )
        if with_mask
        else None
    )

    # causal follows production: forced to 0 iff a custom mask is present.
    causal = 0 if with_mask else 1

    @functools.partial(jax.jit, static_argnames=("sm_scale",))
    def attn(q, k, v, kv_cache, kv_lens, page_indices, cu_q_lens, cu_kv_lens, dist, sm_scale):
        return ragged_paged_attention(
            q,
            k,
            v,
            kv_cache,
            kv_lens,
            page_indices,
            cu_q_lens,
            cu_kv_lens,
            dist,
            custom_mask=mask,
            causal=causal,
            sm_scale=sm_scale,
            vmem_limit_bytes=get_vmem_limit(),
        )

    args = (q, k, v, kv_cache, kv_lens, page_indices, cu_q_lens, cu_kv_lens, distribution)
    bound = functools.partial(attn, *args, sm_scale=head_dim**-0.5)
    jax.block_until_ready(bound())  # compile outside the trace

    task = f"verify_q{draft_token_num}_{'mask' if with_mask else 'nomask'}"
    times = multiple_iteration_timeit_from_trace(
        lambda: bound(), lambda: (), task=task, tries=tries
    )
    if not times:
        raise RuntimeError(f"no device timings for {task}; the trace regex found nothing")
    return statistics.median(times)


# --------------------------------------------------------------------------
# B. the speculative kernels chain drafting never runs
# --------------------------------------------------------------------------
def bench_tree_kernels() -> dict[str, float]:
    """Wall-clock ms for each tree-only Pallas kernel, at the repo's own fixtures.

    NOTE THE UNIT. These reuse the repo's own benchmark functions, which time
    with time.perf_counter, so the numbers include host dispatch. Part A above is
    device time from the profiler. The two are NOT summable, which is why the
    break-even table uses verify only and treats these as "and there is more".

    Fixed shapes, so this is a magnitude, not a curve. Chain drafting runs none
    of them, so whatever they cost is pure tree overhead per iteration.
    """
    # bench_speculative_kernels imports CustomTestCase/is_in_ci for a test class
    # it also defines; that import pulls in flax, and some hosted TPU images ship
    # a flax newer than their jax (ImportError on jax._src.core.mutable_array).
    # The three benchmark functions need neither name, so stub the module rather
    # than require a working flax to time three Pallas kernels.
    if "sgl_jax.test.test_utils" not in sys.modules:
        stub = types.ModuleType("sgl_jax.test.test_utils")
        stub.CustomTestCase = object
        stub.is_in_ci = lambda: False
        sys.modules["sgl_jax.test.test_utils"] = stub

    from benchmark.kernels.speculative import bench_speculative_kernels as bsk

    out = {}
    for name, fn in (
        ("build_eagle_tree_structure", bsk.benchmark_build_eagle_tree_structure),
        ("verify_tree_greedy", bsk.benchmark_verify_tree_greedy),
        ("tree_speculative_sampling", bsk.benchmark_tree_speculative_sampling),
    ):
        try:
            out[name] = fn() * 1000.0  # the module returns seconds
        except Exception as exc:
            print(f"  ! {name} failed: {type(exc).__name__}: {exc}")
    return out


# --------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--bs", type=int, default=16, help="batch size")
    ap.add_argument("--kv-len", type=int, default=2048, help="prefix + draft tokens per sequence")
    ap.add_argument(
        "--chain-draft-tokens",
        type=int,
        default=4,
        help="num_draft_tokens for the chain baseline (the shipped e2e config uses 4)",
    )
    ap.add_argument(
        "--tree-draft-tokens",
        type=int,
        nargs="+",
        default=[8, 16, 32, 64],
        help="num_draft_tokens values to try for the tree",
    )
    ap.add_argument("--q-heads", type=int, default=32)
    ap.add_argument("--kv-heads", type=int, default=8)
    ap.add_argument("--head-dim", type=int, default=128)
    ap.add_argument("--page-size", type=int, default=128)
    ap.add_argument("--tries", type=int, default=15)
    args = ap.parse_args()

    check_env()

    common = dict(
        batch_size=args.bs,
        kv_len=args.kv_len,
        q_head_num=args.q_heads,
        kv_head_num=args.kv_heads,
        head_dim=args.head_dim,
        page_size=args.page_size,
        tries=args.tries,
    )

    print("=" * 72)
    print(
        f"A. target-verify attention   bs={args.bs} kv_len={args.kv_len} "
        f"q_heads={args.q_heads} kv_heads={args.kv_heads} hd={args.head_dim} bf16"
    )
    print("=" * 72)

    chain_ms = bench_verify(
        draft_token_num=args.chain_draft_tokens, with_mask=False, **common
    )
    print(
        f"  chain  q={args.chain_draft_tokens:<3} no mask, causal=1   "
        f"{chain_ms:8.4f} ms   <- baseline"
    )

    # Measure the baseline a second time. Any spread is pure run-to-run noise,
    # and it sets the bar for how large a ratio has to be to mean anything. This
    # is a designed probe, not a spare number: on a first run the same config
    # timed twice differed by 8%, which is larger than most of the ratios below.
    chain_ms2 = bench_verify(
        draft_token_num=args.chain_draft_tokens, with_mask=False, **common
    )
    noise = abs(chain_ms2 - chain_ms) / ((chain_ms + chain_ms2) / 2)
    print(
        f"  chain  q={args.chain_draft_tokens:<3} repeat              "
        f"{chain_ms2:8.4f} ms   noise floor {noise * 100:.1f}%"
    )
    if noise > 0.03:
        print("         ^ raise --tries; ratios below this are not measurements")
    print()

    tree_rows = []
    for n in args.tree_draft_tokens:
        try:
            ms = bench_verify(draft_token_num=n, with_mask=True, **common)
        except Exception as exc:
            print(f"  tree   q={n:<3} FAILED: {type(exc).__name__}: {exc}")
            continue
        tree_rows.append((n, ms))
        ratio = ms / chain_ms
        verdict = "within noise" if abs(ratio - 1.0) <= noise else ""
        print(
            f"  tree   q={n:<3} tree mask, causal=0 {ms:8.4f} ms   "
            f"{ratio:6.2f}x chain  {verdict}"
        )

    # Same q, mask on/off: how much of the tree's verify cost is the mask itself
    # rather than the extra rows. Not production-shaped, but it says whether the
    # remaining gap is worth attacking.
    print()
    print("  isolation (same q, mask on vs off):")
    for n in args.tree_draft_tokens[:2]:
        try:
            a = bench_verify(draft_token_num=n, with_mask=False, **common)
            b = bench_verify(draft_token_num=n, with_mask=True, **common)
            print(f"    q={n:<3} nomask {a:8.4f} ms | mask {b:8.4f} ms | mask costs {b / a:5.2f}x")
        except Exception as exc:
            print(f"    q={n:<3} FAILED: {type(exc).__name__}: {exc}")

    print()
    print("=" * 72)
    print("B. tree-only kernels (chain drafting runs none of these)")
    print("=" * 72)
    tree_kernels = bench_tree_kernels()
    for name, ms in tree_kernels.items():
        print(f"  {name:<34} {ms:8.4f} ms")
    tree_kernel_total = sum(tree_kernels.values())
    if tree_kernels:
        print(f"  {'total':<34} {tree_kernel_total:8.4f} ms   (at the fixture shape, not swept)")
    print("  ^ wall clock, includes host dispatch. Part A is device time. Do not add them;")
    print("    read this as 'and the tree pays this on top', not as a term in the ratio.")

    print()
    print("=" * 72)
    print("break-even")
    print("=" * 72)
    print("  tau is the accept length: tokens committed per verify step, floor 1.0.")
    print("  The tree pays off only when  tau_tree / tau_chain  exceeds the cost ratio below.")
    print()
    print(f"  {'tree q':>8}  {'verify ms':>11}  {'cost ratio':>11}  {'tau_tree must reach':>21}")
    for n, ms in tree_rows:
        ratio = ms / chain_ms
        need = "no gain needed" if abs(ratio - 1.0) <= noise else "tau_chain x %.2f" % ratio
        print(f"  {n:>8}  {ms:>11.4f}  {ratio:>10.2f}x  {need:>21}")
    print()
    print("  Verify attention only. This is a LOWER bound on the cost ratio -- everything")
    print("  omitted works against the tree:")
    print("    - the three tree kernels in section B, which chain drafting never runs")
    print("    - the draft model forward on bs*topk rows instead of bs*1")
    print("    - host-side tree bookkeeping (still a Python loop)")
    print("    - the overlap scheduler, which topk>1 forces off")
    print()
    print("  So a tree that already fails this break-even fails for real. One that passes")
    print("  it has not yet been shown to pay off.")
    print()
    print("  Next: measure tau_chain (Exp 1) and bound tau_tree (Exp 2). If a realistic")
    print("  tau_tree/tau_chain cannot clear the ratio above, topk>1 is not worth building.")


if __name__ == "__main__":
    main()
