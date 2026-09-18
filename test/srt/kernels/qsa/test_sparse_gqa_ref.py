"""CPU tests for the sparse GQA attention reference.

The reference is what the Pallas kernel is checked against, so it gets its own
tests: an oracle nobody has verified is worth nothing.
"""

import os
import unittest

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax.numpy as jnp
import numpy as np

from sgl_jax.srt.kernels.qsa.ref import selected_tokens_ref, sparse_gqa_attention_ref
from sgl_jax.test.test_utils import CustomTestCase

RATIO = 4
PAGE_SIZE = 8
HEAD_DIM = 16
SEED = 7


def _softmax(x):
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)


class TestSelectedTokens(CustomTestCase):
    def test_blocks_expand_and_the_open_group_follows(self):
        """pos=10 sits inside an incomplete group, so 8..10 come along as the
        tail. pos=11 closes it, and the group competes in the top-k instead."""
        self.assertEqual(
            selected_tokens_ref([0, 1], 10, compress_ratio=RATIO),
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        )
        self.assertEqual(
            selected_tokens_ref([0, 2], 11, compress_ratio=RATIO),
            [0, 1, 2, 3, 8, 9, 10, 11],
        )

    def test_padding_is_ignored(self):
        """-1 block ids expand to nothing rather than to tokens around zero."""
        self.assertEqual(selected_tokens_ref([0, -1, -1], 7, compress_ratio=RATIO), [0, 1, 2, 3])

    def test_nothing_reaches_past_the_query(self):
        """A block whose tokens run past the query keeps only the visible ones."""
        rng = np.random.default_rng(SEED)
        for pos in range(24):
            n_visible = (pos + 1) // RATIO
            blocks = rng.choice(max(n_visible, 1), size=2) if n_visible else [-1, -1]
            tokens = selected_tokens_ref(blocks, pos, compress_ratio=RATIO)
            if tokens:
                self.assertLessEqual(max(tokens), pos, f"position {pos}")

    def test_a_query_at_a_group_boundary_has_no_tail(self):
        """When the query closes its group there is no open tail, so the selected
        blocks are the whole answer."""
        # pos = 4k - 1 closes group k-1, so (pos + 1) % ratio == 0.
        for pos in (3, 7, 11, 15):
            tokens = selected_tokens_ref([], pos, compress_ratio=RATIO)
            self.assertEqual(tokens, [], f"position {pos}")


class TestSparseGQAAttentionRef(CustomTestCase):
    def _case(self, n_q_heads, n_kv_heads, t_count=3, n_pages=4):
        rng = np.random.default_rng(SEED)
        k = rng.standard_normal((n_pages, PAGE_SIZE, n_kv_heads, HEAD_DIM)).astype(np.float32)
        v = rng.standard_normal((n_pages, PAGE_SIZE, n_kv_heads, HEAD_DIM)).astype(np.float32)
        q = rng.standard_normal((t_count, n_q_heads, HEAD_DIM)).astype(np.float32)
        return q, k, v

    def test_matches_dense_attention_over_the_same_tokens(self):
        """Given the token set it selects, the reference is plain softmax attention
        -- it adds no masking or scaling of its own."""
        n_q, n_kv, t_count = 4, 1, 3
        q, k, v = self._case(n_q, n_kv, t_count)
        page_table = jnp.asarray([[0, 1, 2, 3]], jnp.int32)
        token_to_req = jnp.zeros((t_count,), jnp.int32)
        positions = jnp.asarray([10, 11, 17], jnp.int32)
        block_ids = jnp.asarray([[0, 1], [0, 2], [1, 3]], jnp.int32)
        sm_scale = HEAD_DIM**-0.5

        got = np.asarray(
            sparse_gqa_attention_ref(
                jnp.asarray(q),
                block_ids,
                positions,
                jnp.asarray(k),
                jnp.asarray(v),
                page_table,
                token_to_req,
                compress_ratio=RATIO,
                sm_scale=sm_scale,
            )
        )

        flat_k = k.reshape(-1, n_kv, HEAD_DIM)
        flat_v = v.reshape(-1, n_kv, HEAD_DIM)
        for t in range(t_count):
            tokens = selected_tokens_ref(
                np.asarray(block_ids)[t], int(positions[t]), compress_ratio=RATIO
            )
            kk = np.repeat(flat_k[tokens], n_q // n_kv, axis=1)  # [N, H, D]
            vv = np.repeat(flat_v[tokens], n_q // n_kv, axis=1)
            scores = np.einsum("hd,nhd->hn", q[t], kk) * sm_scale
            want = np.einsum("hn,nhd->hd", _softmax(scores), vv)
            np.testing.assert_allclose(got[t], want, atol=1e-5, rtol=1e-5)

    def test_query_heads_read_their_own_kv_head(self):
        """With 2 KV heads and 4 query heads, heads 0-1 must read KV head 0 and
        heads 2-3 KV head 1. Zeroing one KV head may only move its own queries."""
        n_q, n_kv, t_count = 4, 2, 2
        q, k, v = self._case(n_q, n_kv, t_count)
        page_table = jnp.asarray([[0, 1, 2, 3]], jnp.int32)
        token_to_req = jnp.zeros((t_count,), jnp.int32)
        positions = jnp.asarray([11, 15], jnp.int32)
        block_ids = jnp.asarray([[0, 1], [1, 2]], jnp.int32)
        common = dict(compress_ratio=RATIO, sm_scale=HEAD_DIM**-0.5)

        base = np.asarray(
            sparse_gqa_attention_ref(
                jnp.asarray(q),
                block_ids,
                positions,
                jnp.asarray(k),
                jnp.asarray(v),
                page_table,
                token_to_req,
                **common,
            )
        )
        v_zeroed = v.copy()
        v_zeroed[:, :, 1, :] = 0.0  # kill KV head 1
        perturbed = np.asarray(
            sparse_gqa_attention_ref(
                jnp.asarray(q),
                block_ids,
                positions,
                jnp.asarray(k),
                jnp.asarray(v_zeroed),
                page_table,
                token_to_req,
                **common,
            )
        )

        np.testing.assert_allclose(base[:, :2], perturbed[:, :2], atol=0, rtol=0)
        self.assertFalse(np.allclose(base[:, 2:], perturbed[:, 2:]))

    def test_rejects_a_head_count_that_is_not_a_multiple(self):
        """Query heads must divide evenly among KV heads, or the GQA mapping is
        undefined."""
        q, k, v = self._case(3, 2, 1)
        with self.assertRaisesRegex(ValueError, "multiple"):
            sparse_gqa_attention_ref(
                jnp.asarray(q),
                jnp.asarray([[0]], jnp.int32),
                jnp.asarray([7], jnp.int32),
                jnp.asarray(k),
                jnp.asarray(v),
                jnp.asarray([[0, 1, 2, 3]], jnp.int32),
                jnp.zeros((1,), jnp.int32),
                compress_ratio=RATIO,
                sm_scale=HEAD_DIM**-0.5,
            )


if __name__ == "__main__":
    unittest.main()
