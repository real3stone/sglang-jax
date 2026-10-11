"""Host-resident PLE table: 320,001,446 rows x 160, 95.4 GiB in bf16 or
47.7 GiB in the FP8 checkpoint's float8_e4m3fn with one per-table scale.

It does not fit in v6e HBM, and XLA:TPU cannot gather from a pinned_host
array, so the host gathers 16 rows per token and only [T, 16*160] crosses to
the device. The gather is memory-latency-bound, so _thread_count scales the
thread count with the row count.
"""

from __future__ import annotations

import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import ml_dtypes
import numpy as np

from sgl_jax.srt.layers.ngram_embedding import NGramHashParams, build_hash_params

logger = logging.getLogger(__name__)

_ROWS_PER_THREAD = 1024  # 128 rows -> 1t, 4,096 -> 4t, 131,072 -> 32t
_MAX_THREADS = 32  # past this the box fights itself

CACHE_FORMAT_VERSION = 2

# Storage per checkpoint dtype: the row bytes as unsigned integers, since the
# gather is pure byte movement.
_STORAGE = {"bfloat16": np.uint16, "float8_e4m3fn": np.uint8}


def _thread_count(rows: int, cap: int | None = None) -> int:
    cap = _MAX_THREADS if cap is None else cap
    return max(1, min(cap, rows // _ROWS_PER_THREAD))


@dataclass(frozen=True)
class ShardPlacement:
    """Where checkpoint shard ``index`` lands in the full table."""

    index: int
    start: int
    rows: int


def shard_placements(total_rows: int, split_parts: int) -> list[ShardPlacement]:
    """Row range of every ``ngram_embedding.shard_{i}.weight``.

    ``total_rows`` is the padded height: the exporter shards after padding.
    """
    if split_parts <= 0:
        raise ValueError(f"split_ngram_parts must be positive, got {split_parts}")
    shard_size = (total_rows + split_parts - 1) // split_parts
    out = []
    for i in range(split_parts):
        start = i * shard_size
        rows = max(0, min(shard_size, total_rows - start))
        if rows:
            out.append(ShardPlacement(index=i, start=start, rows=rows))
    return out


class NGramTable:
    """data [padded_rows, dim] as raw row bytes: uint16 for bfloat16 rows,
    uint8 for float8_e4m3fn rows. :meth:`gather` always returns bfloat16; FP8
    rows are dequantized through a 256-entry table with the scale folded in."""

    def __init__(
        self,
        params: NGramHashParams,
        dim: int,
        padded_rows: int | None = None,
        dtype: str = "bfloat16",
        scale: float = 1.0,
    ):
        if dtype not in _STORAGE:
            raise ValueError(f"unsupported N-gram table dtype {dtype!r}")
        self.params = params
        self.dim = int(dim)
        self.dtype = dtype
        self.scale = float(scale)
        # fp8 bit pattern -> bf16 bit pattern of value * scale
        self._dequant = (
            (
                np.arange(256, dtype=np.uint8).view(ml_dtypes.float8_e4m3fn).astype(np.float32)
                * np.float32(self.scale)
            )
            .astype(ml_dtypes.bfloat16)
            .view(np.uint16)
            if dtype == "float8_e4m3fn"
            else None
        )
        self.rows = int(params.total_vocab_size)
        self.padded_rows = int(padded_rows) if padded_rows else self.rows
        if self.padded_rows < self.rows:
            raise ValueError(
                f"padded_rows {self.padded_rows} is smaller than the table's {self.rows}"
            )
        self.data = np.zeros((self.padded_rows, self.dim), dtype=_STORAGE[dtype])
        self._pool: ThreadPoolExecutor | None = None
        logger.info(
            "N-gram table allocated: %s rows x %d = %.1f GiB host RAM",
            f"{self.padded_rows:,}",
            self.dim,
            self.data.nbytes / 2**30,
        )

    # -- gather --------------------------------------------------------------

    def gather(self, ids: np.ndarray, out: np.ndarray | None = None) -> np.ndarray:
        """ids [T, HEADS] -> [T, HEADS*dim] bf16, the batch's ple_embeddings.

        bf16 not uint16: device_array would otherwise put an integer array on
        the device and key_proj would matmul the bit patterns. ``out``, if
        given, is [T, HEADS, dim] uint16 and is reused across steps.
        """
        ids = np.asarray(ids)
        if ids.ndim != 2 or ids.shape[1] != self.params.ngram_heads:
            raise ValueError(f"ids must be [T, {self.params.ngram_heads}], got {tuple(ids.shape)}")
        num_tokens, heads = ids.shape  # T, HEADS
        if out is None:
            out = np.empty((num_tokens, heads, self.dim), np.uint16)  # [T, HEADS, dim]
        flat_ids = ids.reshape(-1)  # [T*HEADS]
        flat_out = out.reshape(-1, self.dim)  # [T*HEADS, dim]

        def gather_rows(lo: int, hi: int) -> None:
            if self._dequant is None:
                np.take(self.data, flat_ids[lo:hi], axis=0, out=flat_out[lo:hi])
            else:
                np.take(
                    self._dequant, np.take(self.data, flat_ids[lo:hi], axis=0), out=flat_out[lo:hi]
                )

        nthreads = _thread_count(flat_ids.shape[0])
        if nthreads == 1:
            gather_rows(0, flat_ids.shape[0])
        else:
            if self._pool is None:
                self._pool = ThreadPoolExecutor(
                    max_workers=_MAX_THREADS, thread_name_prefix="ngram-gather"
                )
            bounds = np.linspace(0, flat_ids.shape[0], nthreads + 1).astype(int)
            list(  # np.take drops the GIL, so these actually overlap
                self._pool.map(lambda k: gather_rows(bounds[k], bounds[k + 1]), range(nthreads))
            )
        return out.reshape(num_tokens, heads * self.dim).view(ml_dtypes.bfloat16)

    # -- checkpoint ----------------------------------------------------------

    def load_shard(self, placement: ShardPlacement, weight: np.ndarray) -> None:
        """Copy one checkpoint shard into its row range."""
        if weight.shape != (placement.rows, self.dim):
            raise ValueError(
                f"shard {placement.index}: expected {(placement.rows, self.dim)}, "
                f"got {tuple(weight.shape)}"
            )
        storage = _STORAGE[self.dtype]
        if weight.dtype.itemsize != np.dtype(storage).itemsize:
            raise ValueError(
                f"shard {placement.index}: {weight.dtype} rows in a {self.dtype} table"
            )
        dst = self.data[placement.start : placement.start + placement.rows]  # [rows, dim]
        dst[:] = weight.view(storage)

    @classmethod
    def from_safetensors(
        cls,
        weight_files: dict[str, str | Path],
        config,
        *,
        ple_dense_layer_id: int = 0,
        prefix: str = "model.language_model.layers.1.ple.ple_embedding",
    ) -> NGramTable:
        """Stream the shards into RAM. ``weight_files`` is the checkpoint
        index's weight_map (tensor name -> safetensors file)."""
        from safetensors import safe_open

        params = build_hash_params(
            ngram_size=config.ngram_size,
            heads_per_ngram=config.heads_per_ngram,
            vocab_size=config.vocab_size,
            ngram_vocab_size_base=config.ngram_vocab_size_base,
            eos_token_id=_eos_token_id(config),
            ple_dense_layer_id=ple_dense_layer_id,
        )
        divisor = int(config.make_ngram_vocab_size_divisible_by)  # 128
        padded = ((params.total_vocab_size + divisor - 1) // divisor) * divisor  # 320,001,536
        dim = int(config.ple_embed_dim) // params.ngram_heads  # 2560 // 16 = 160
        # The FP8 checkpoint ships one scale for the whole table.
        scale_name = f"{prefix}.ngram_embedding.weight_scale"
        if scale_name in weight_files:
            scale = _read_raw(weight_files[scale_name], scale_name)
            scale = float(scale.view(ml_dtypes.bfloat16).astype(np.float32).reshape(-1)[0])
            table = cls(params, dim, padded_rows=padded, dtype="float8_e4m3fn", scale=scale)
        else:
            table = cls(params, dim, padded_rows=padded)

        # Group by file so each one is opened once, in index order.
        placements = shard_placements(table.padded_rows, int(config.split_ngram_parts))
        by_file: dict[str, list[ShardPlacement]] = {}
        for p in placements:
            name = f"{prefix}.ngram_embedding.shard_{p.index}.weight"
            if name not in weight_files:
                raise KeyError(f"checkpoint is missing {name}")
            by_file.setdefault(str(weight_files[name]), []).append(p)

        done = 0
        for path, group in by_file.items():
            if table.dtype == "float8_e4m3fn":
                for p in group:
                    name = f"{prefix}.ngram_embedding.shard_{p.index}.weight"
                    table.load_shard(p, _read_raw(path, name))
                    done += 1
            else:
                with safe_open(path, framework="np") as f:
                    for p in group:
                        table.load_shard(
                            p, f.get_tensor(f"{prefix}.ngram_embedding.shard_{p.index}.weight")
                        )
                        done += 1
            logger.info("N-gram table: %d/%d shards", done, len(placements))

        table.verify_against_checkpoint(weight_files, prefix=prefix)
        return table

    def verify_against_checkpoint(self, weight_files, *, prefix: str) -> None:
        """Check the derived hash layout against the exporter's buffers. A wrong
        multiplier would silently read valid but wrong rows."""
        from safetensors import safe_open

        expected = {
            "layer_multipliers": self.params.multipliers,
            "ngram_heads_vocab_sizes": self.params.sizes,
            "ngram_heads_offsets": self.params.offsets,
        }
        for short, derived in expected.items():
            name = f"{prefix}.{short}"
            path = weight_files.get(name)
            if path is None:
                logger.warning("checkpoint has no %s; keeping the derived values", name)
                continue
            with safe_open(str(path), framework="np") as f:
                shipped = np.asarray(f.get_tensor(name), dtype=np.int64)
            if not np.array_equal(shipped, derived):
                bad = int(np.argmax(shipped != derived))
                raise ValueError(
                    f"{name} does not match the derived hash layout at index {bad}: "
                    f"checkpoint {shipped[bad]} vs derived {derived[bad]}. "
                    "Every N-gram lookup would silently read the wrong row."
                )

    # -- local cache ---------------------------------------------------------

    _STREAM_ROWS = 1 << 20  # ~320 MiB per chunk at the released row width

    def write_to(self, stream) -> int:
        """Raw rows out, chunked; pair with read_into + metadata. A stream because
        v6e-1's disk is smaller than the table."""
        written = 0
        for lo in range(0, self.padded_rows, self._STREAM_ROWS):
            block = self.data[lo : lo + self._STREAM_ROWS].tobytes()
            stream.write(block)
            written += len(block)
        return written

    def read_into(self, stream) -> None:
        """Fill the table from a stream written by :meth:`write_to`."""
        view = memoryview(self.data.reshape(-1).view(np.uint8))
        offset = 0
        while offset < len(view):
            chunk = stream.read(min(1 << 26, len(view) - offset))
            if not chunk:
                break
            view[offset : offset + len(chunk)] = chunk
            offset += len(chunk)
        if offset != len(view):
            raise ValueError(f"stream ended after {offset} bytes, expected {len(view)}")

    def metadata(self) -> dict:
        return {
            "version": CACHE_FORMAT_VERSION,
            "rows": self.rows,
            "padded_rows": self.padded_rows,
            "dim": self.dim,
            "dtype": self.dtype,
            "scale": self.scale,
            "ngram_size": self.params.ngram_size,
            "heads_per_ngram": self.params.heads_per_ngram,
            "eos_token_id": self.params.eos_token_id,
            "multipliers": self.params.multipliers.tolist(),
            "sizes": self.params.sizes.tolist(),
            "offsets": self.params.offsets.tolist(),
        }

    @classmethod
    def from_metadata(cls, meta: dict) -> NGramTable:
        """Allocated but empty; fill with :meth:`read_into`. Version 1 caches
        are bfloat16 tables without a scale."""
        if meta["version"] not in (1, CACHE_FORMAT_VERSION):
            raise ValueError(
                f"N-gram table cache is version {meta['version']}, "
                f"this build reads 1 and {CACHE_FORMAT_VERSION}"
            )
        params = NGramHashParams(
            multipliers=np.array(meta["multipliers"], np.int64),
            sizes=np.array(meta["sizes"], np.int64),
            offsets=np.array(meta["offsets"], np.int64),
            total_vocab_size=meta["rows"],
            heads_per_ngram=meta["heads_per_ngram"],
            ngram_size=meta["ngram_size"],
            eos_token_id=meta["eos_token_id"],
        )
        return cls(
            params,
            meta["dim"],
            padded_rows=meta["padded_rows"],
            dtype=meta.get("dtype", "bfloat16"),
            scale=meta.get("scale", 1.0),
        )

    def save_cache(self, directory: str | Path) -> Path:
        """Flat file + json sidecar, restored with one sequential read."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "ngram_table.json").write_text(json.dumps(self.metadata()))
        with open(directory / "ngram_table.bin", "wb") as f:
            self.write_to(f)
        return directory

    @classmethod
    def from_cache(cls, directory: str | Path) -> NGramTable:
        directory = Path(directory)
        meta = json.loads((directory / "ngram_table.json").read_text())
        table = cls.from_metadata(meta)
        path = directory / "ngram_table.bin"
        storage = _STORAGE[table.dtype]
        want = table.padded_rows * table.dim * np.dtype(storage).itemsize
        got = os.path.getsize(path)
        if got != want:
            raise ValueError(f"{path} is {got} bytes, expected {want}")
        # mmap: pages load lazily and the OS can evict them.
        table.data = np.memmap(path, dtype=storage, mode="r").reshape(table.padded_rows, table.dim)
        _madvise_random(table.data)
        return table


_MADV_RANDOM = 1


def _madvise_random(array: np.ndarray) -> bool:
    """Reads are 320 B rows scattered over the table, so readahead only wastes
    I/O. Best-effort."""
    import ctypes
    import ctypes.util

    try:
        libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
        addr = array.ctypes.data
        page = os.sysconf("SC_PAGESIZE")
        base = addr - (addr % page)
        rc = libc.madvise(
            ctypes.c_void_p(base),
            ctypes.c_size_t(array.nbytes + (addr - base)),
            ctypes.c_int(_MADV_RANDOM),
        )
    except Exception as exc:  # noqa: BLE001 - advisory only
        logger.warning("N-gram table: madvise(MADV_RANDOM) unavailable (%s)", exc)
        return False
    if rc != 0:
        logger.warning("N-gram table: madvise(MADV_RANDOM) failed")
        return False
    return True


# safetensors dtype -> storage holding its raw bytes
_RAW_DTYPES = {"BF16": np.uint16, "F8_E4M3": np.uint8, "F32": np.float32, "I64": np.int64}


def _read_raw(path: str | Path, name: str) -> np.ndarray:
    """One tensor's bytes from a safetensors file, as its storage integers.

    safetensors' NumPy backend cannot return float8 arrays, so FP8 rows and
    their bf16 scale are read straight from the file's header and offsets.
    """
    with open(path, "rb") as f:
        (header_len,) = np.frombuffer(f.read(8), dtype="<u8")
        header = json.loads(f.read(int(header_len)))
        info = header.get(name)
        if info is None:
            raise KeyError(f"{path} has no tensor {name}")
        if info["dtype"] not in _RAW_DTYPES:
            raise ValueError(f"{name}: unsupported safetensors dtype {info['dtype']}")
        start, end = info["data_offsets"]
        f.seek(8 + int(header_len) + start)
        data = np.frombuffer(f.read(end - start), dtype=_RAW_DTYPES[info["dtype"]])
    return data.reshape(info["shape"])


def _eos_token_id(config) -> int:
    eos = getattr(config, "eos_token_id", None)
    if isinstance(eos, (list, tuple)):
        eos = eos[0]
    if eos is None:
        raise ValueError("config has no eos_token_id; the N-gram hash needs it as a barrier")
    return int(eos)


# -- process-global handle: one table per process ----------------------------

_TABLE: NGramTable | None = None


def set_ngram_table(table: NGramTable | None) -> None:
    """Install the table for the scheduler's host-side lookup. Call once at
    weight load; ``None`` uninstalls it."""
    global _TABLE
    _TABLE = table


def get_ngram_table() -> NGramTable | None:
    """The installed table, or None for every model without a PLE layer."""
    return _TABLE


__all__ = [
    "NGramTable",
    "ShardPlacement",
    "get_ngram_table",
    "set_ngram_table",
    "shard_placements",
]
