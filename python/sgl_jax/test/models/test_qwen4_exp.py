"""Qwen3.8-Flash-Next assembly, on CPU.

Shape and structure only -- numbers need the real checkpoint. What is pinned
here is the four things the backbone does differently from Qwen3.5, each of
which is silent if wrong: which layers get which block, that every block is
wrapped in its own hyper connection, that the streams widen exactly once, and
that the mapping table names parameters the model actually has.
"""

from __future__ import annotations

import importlib.util
import os
import types
import unittest

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from jax.sharding import AxisType, Mesh

from sgl_jax.srt.configs.qwen4_exp import Qwen4ExpConfig
from sgl_jax.srt.layers.embeddings import MRotaryEmbedding, RotaryEmbedding
from sgl_jax.srt.models.qwen4_exp import (
    Qwen4ExpAttention,
    Qwen4ExpModel,
    _create_qwen4_exp_weight_mappings,
)
from sgl_jax.test.test_utils import CustomTestCase

NUM_LAYERS = 8
INTERVAL = 4
PLE_LAYER_1BASED = 2


def _mesh():
    return Mesh(
        np.array(jax.devices())[:1].reshape(1, 1),
        axis_names=("data", "tensor"),
        axis_types=(AxisType.Explicit, AxisType.Explicit),
    )


# The section widths a real checkpoint ships, scaled to this test's head_dim:
# they have to sum to rotary_dim // 2.
MROPE_SECTION = [3, 3, 2]


def _config(*, num_layers=NUM_LAYERS, ple=False, mrope=False):
    text = dict(
        num_hidden_layers=num_layers,
        full_attention_interval=INTERVAL,
        hidden_size=256,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=64,
        intermediate_size=512,
        vocab_size=512,
        num_experts=8,
        moe_intermediate_size=256,
        shared_expert_intermediate_size=256,
        num_experts_per_tok=2,
        indexer_budget=2048,
        indexer_compress_ratio=4,
        indexer_head_dim=128,
        indexer_n_heads=4,
        indexer_kv_heads=1,
    )
    if ple:
        text["ple_layer_ids"] = [PLE_LAYER_1BASED]
    if mrope:
        # A real checkpoint nests RoPE under rope_parameters; the config
        # class flattens it into rope_scaling.
        text["rope_parameters"] = dict(
            rope_type="default",
            mrope_section=MROPE_SECTION,
            mrope_interleaved=True,
            rope_theta=10000000,
            partial_rotary_factor=0.25,
        )
    return Qwen4ExpConfig(text_config=text)


def _model(cfg, mesh):
    with jax.set_mesh(mesh):
        return Qwen4ExpModel(cfg, mesh)


class TestBackboneStructure(CustomTestCase):
    def test_layer_types_follow_the_interval(self):
        """Every fourth layer is full attention and the rest are GDN."""
        cfg = _config()
        model = _model(cfg, _mesh())

        full = [i for i, layer in enumerate(model.layers) if layer.is_full_attn]
        self.assertEqual(full, list(cfg.text_config.full_attention_layer_ids))
        self.assertEqual(len(full), NUM_LAYERS // INTERVAL)
        self.assertTrue(all(layer.ple is None for layer in model.layers))

    @unittest.skipUnless(
        importlib.util.find_spec("sgl_jax.srt.layers.ngram_embedding"),
        "the N-gram embedding module is not in the tree yet",
    )
    def test_the_ngram_layer_is_a_gdn_layer(self):
        """``ple_layer_ids`` is 1-based, matching the checkpoint's numbering,
        and the layer it names is an ordinary GDN layer that happens to carry
        the module -- not a type of its own."""
        model = _model(_config(ple=True), _mesh())
        ple = [i for i, layer in enumerate(model.layers) if layer.ple is not None]
        self.assertEqual(ple, [PLE_LAYER_1BASED - 1])
        self.assertFalse(model.layers[ple[0]].is_full_attn)

    def test_every_block_gets_its_own_hyper_connection(self):
        """Two per layer, because a layer has an attention slot and an MLP slot
        whichever family fills the first one."""
        model = _model(_config(), _mesh())
        for i, layer in enumerate(model.layers):
            self.assertTrue(layer.attn_hyper_connection.use_combine, f"layer {i}")
            self.assertTrue(layer.mlp_hyper_connection.use_combine, f"layer {i}")

    def test_the_hyper_connections_are_the_only_normalization(self):
        """Qwen3.5's three RMSNorms are absorbed into the mix, so neither the
        layers nor the model may still hold one."""
        model = _model(_config(), _mesh())
        self.assertFalse(hasattr(model, "norm"))
        for i, layer in enumerate(model.layers):
            self.assertFalse(hasattr(layer, "input_layernorm"), f"layer {i}")
            self.assertFalse(hasattr(layer, "post_attention_layernorm"), f"layer {i}")

        # The mixer reads the streams down and never writes back, so it builds
        # no injection weights.
        self.assertFalse(model.hyper_connection_mixer.use_combine)

    def test_the_streams_widen_once(self):
        """The embedding is one stream wide and the layers carry hc_count of
        them; widening is idempotent so only the first layer pays it."""
        cfg = _config()
        text = cfg.text_config
        model = _model(cfg, _mesh())
        narrow = jnp.zeros((3, text.hidden_size))
        wide = model.layers[0]._to_streams(narrow)

        self.assertEqual(wide.shape, (3, text.hc_count * text.hidden_size))
        self.assertEqual(model.layers[1]._to_streams(wide).shape, wide.shape)
        # Repeated, not zero-padded: every stream starts as the embedding.
        ones = jnp.ones((3, text.hidden_size))
        np.testing.assert_array_equal(
            np.asarray(model.layers[0]._to_streams(ones)), np.ones((3, wide.shape[-1]))
        )
        with self.assertRaises(ValueError):
            model.layers[0]._to_streams(jnp.zeros((3, text.hidden_size + 1)))


class TestRotary(CustomTestCase):
    def test_the_attention_follows_the_checkpoint_into_mrope(self):
        """A checkpoint that ships mrope_section gets the multimodal rotary.
        Reading only the flat ``rope_scaling`` key says None for both, because
        the real layout nests under ``rope_parameters``."""
        mesh = _mesh()
        with jax.set_mesh(mesh):
            with_mrope = Qwen4ExpAttention(_config(mrope=True), mesh, layer_id=INTERVAL - 1)
            without = Qwen4ExpAttention(_config(), mesh, layer_id=INTERVAL - 1)

        self.assertIsInstance(with_mrope.rotary_emb, MRotaryEmbedding)
        self.assertIsInstance(without.rotary_emb, RotaryEmbedding)
        self.assertNotIsInstance(without.rotary_emb, MRotaryEmbedding)

    def test_the_indexer_gets_a_rotary_sized_to_its_own_slice(self):
        """``QSAIndexer`` hands the rotary only the leading rotary_dim of its
        narrower head, so it needs head_size == rotary_dim -- the attention's
        own rotary is head_dim wide and would be rejected. The indexer's
        positions are group indices, so it is never the multimodal variant."""
        cfg = _config(mrope=True)
        text = cfg.text_config
        mesh = _mesh()
        with jax.set_mesh(mesh):
            attn = Qwen4ExpAttention(cfg, mesh, layer_id=INTERVAL - 1)

        rotary_dim = int(text.head_dim * float(text.partial_rotary_factor))
        self.assertEqual(attn.indexer_rotary_emb.head_size, rotary_dim)
        self.assertNotIsInstance(attn.indexer_rotary_emb, MRotaryEmbedding)
        self.assertEqual(attn.rotary_emb.head_size, text.head_dim)

        # The mismatch only shows at the call, and only on the path the model
        # actually takes, so drive _indexer_step and record what it hands over.
        seen = {}

        def _project(hidden, positions, rotary_emb):
            seen["rotary"] = rotary_emb
            return jnp.zeros((4, text.indexer_n_heads, text.indexer_head_dim)), jnp.zeros(
                (4, text.indexer_head_dim)
            )

        attn.indexer.project = _project
        attn.indexer.compress_batch = lambda *a, **k: (None, None, None, None)
        forward_batch = types.SimpleNamespace(
            attn_backend=types.SimpleNamespace(
                full_slot={attn.layer_id: 0},
                forward_metadata=types.SimpleNamespace(cu_q_lens=jnp.asarray([0, 4], jnp.int32)),
            ),
            req_pool_indices=jnp.asarray([0], jnp.int32),
            positions=jnp.arange(4, dtype=jnp.int32),
        )
        pool = types.SimpleNamespace(get_open_group_buffer=lambda slot: None)

        with jax.set_mesh(mesh):
            attn._indexer_step(jnp.zeros((4, text.hidden_size)), forward_batch, pool)
        self.assertIs(seen["rotary"], attn.indexer_rotary_emb)


class TestWeightMappings(CustomTestCase):
    def test_new_entries_name_parameters_the_model_has(self):
        """The hyper connections and the indexer are what this table adds over
        Qwen3.5's. A target that resolves to nothing loads nothing and raises
        nothing, so check it against the built module tree."""
        cfg = _config()
        model = _model(cfg, _mesh())
        params = {
            ".".join(str(p) for p in path)
            for path, _ in nnx.to_flat_state(nnx.state(model, nnx.Param))
        }

        mappings, _, _ = _create_qwen4_exp_weight_mappings(cfg)
        prefix = "language_model.model."
        added = [
            m.target_path
            for k, m in mappings.items()
            if "hyper_connection" in k or ".indexer." in k
        ]
        self.assertTrue(added)
        for target in added:
            self.assertTrue(target.startswith(prefix), target)
            self.assertIn(target[len(prefix) :], params, f"{target} names no parameter")

    def test_the_absorbed_norms_are_gone_and_the_table_is_injective(self):
        """Leaving Qwen3.5's norms in would point at modules that no longer
        exist; two sources sharing a target would silently load one twice."""
        cfg = _config()
        mappings, _, _ = _create_qwen4_exp_weight_mappings(cfg)

        self.assertNotIn("model.language_model.norm.weight", mappings)
        for key in mappings:
            self.assertNotIn("input_layernorm", key)
            self.assertNotIn("post_attention_layernorm", key)

        # A pre-fused source splits into several targets, so flatten before
        # counting.
        targets = []
        for mapping in mappings.values():
            target = mapping.target_path
            targets.extend(target if isinstance(target, list) else [target])
        duplicated = {t for t in targets if targets.count(t) > 1}
        self.assertEqual(duplicated, set())


if __name__ == "__main__":
    unittest.main()
