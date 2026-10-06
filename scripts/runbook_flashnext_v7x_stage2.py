"""Qwen3.8-Flash-Next 在 TPU v7x 上的第二轮测试：kernel 真机验证 + MoE 调参，以及打开 N-gram（PLE）
层之后的精度和性能。

和 runbook_flashnext_v7x.py 用同一套环境（前置条件见那份脚本的说明），一次跑完两部分，结束时留下
两个压缩包。

各阶段
  E0  环境检查：runbook 的 C0 那几项，再要求主机可用内存不少于 128 GiB（n-gram 表 95.4 GiB
      常驻主机内存）。不满足就停。

  Part 1：不加载模型权重。任何一项失败都不阻断 Part 2，失败的那一项退回已知可用的配置，接着跑。
  P1  QSA 稀疏 kernel：在 TPU 上跑 test/srt/kernels/qsa/test_sparse_gqa_parity.py（Mosaic 真编译）；
      再按模型的形状（每个设备 3 个 Q 头、1 个 KV 头、head_dim 256、页 64、512 块、block_units 128）
      测每个 query 的耗时，可见块数 16 / 128 / 512，和之前 profile 里的 22.7 µs 放在一起。
      测试或编译失败（每一步最多等 30 分钟）：把工作区的 sparse_gqa_attention.py 还原成
      4c85cce7 的版本，再跑一遍 parity 测试。
  P2  MoE 调参：bench_fused_moe --tune-block-config，覆盖 Part 2 用到的 16 / 64 / 512 / 1024 / 2048
      五个 token 档位，结果存成 moe_tuned.json。调参崩了就不写条目，Part 2 用回退分块；
      有档位没调出条目判 WARN，那几个档位用回退分块。
  P3  调好的分块逐条做数值检查（512 专家、top-10、2560 / 640、ep=8、atol / rtol 5e-2，连同调好的
      bts 一起检查），通过的条目
      临时写进工作区的 tuned_block_configs.py，git diff 存成 p3_tuned_configs.diff。

  Part 2：两次启动。
  S1  PLE 打开启动（上下文 69632、64 个请求）：加载摘要逐字等于
      "consumed=1163, skipped=495, missing=0, unexpected=0"，日志里有 "N-gram table: 128/128 shards"，
      哈希校验没有被跳过；P3 写进了条目时，日志里要有 "Using tuned block config"；再发一个请求，
      能正常出 token、logprob 里没有 NaN。
  S2  MMLU 冒烟：参数和仓库测试 test/srt/test_qwen3_5_models.py 的 _run_mmlu_smoke 完全一致（100 题、
      thinking、max_tokens 32768、seed 17）。高于 0.70 判 PASS；低于 0.30 判 WARN（先查请求是否被
      服务端拒绝）；其余判 FAIL。
  S3  GPQA-Diamond：seed 17 / 23 / 41 / 59 / 97，模型卡推荐的 thinking 采样参数，max_tokens 65536。
      只记录：每个 seed 的分数、中位数、和模型卡 91.7 的差；逐题诊断有被截断（finish=length）的数、
      没抽到答案的数、输出长度分布。每个 seed 跑完就落盘。S1 没过或者 S2 低于 0.30 时不跑：
      模型已经不对，GPQA 没有意义。seed 只用来区分 5 次运行，服务端目前不使用请求里的 seed。
  S4  PLE 打开时的压测：512 入 / 128 出 × 100、8192 入 / 128 出 × 32，并发 8。只记录。随后在预热好
      的服务上抓几步 decode 的 profile（加 --profile-prefill 时连 prefill 一起抓）。
  S5  PLE 关闭启动，参数和第一轮跑通测试相同：同样两组压测，再跑一次 MMLU 冒烟。只记录。
      上下文是 32768，MMLU 的 max_tokens 只能用 30720；和 S2 的差就是 PLE 开 / 关的精度差，
      两边 max_tokens 不同。

  评测都放在子进程里跑，同时盯着服务进程：服务一退出，评测就停下来，当前阶段记 FAIL。
  单个请求最多等 6 小时，超时就记下来、不重发；有请求出错时阶段判 WARN，出错数写进报告。

状态
  PASS 通过；WARN 能跑但有可疑之处；FAIL 失败；未运行：这次没选这个阶段，或者前面的结果已经
  说明它没有意义。
  E0 的 FAIL 是本机环境问题：按列出的问题修好后重跑，这种情况不打包、不用发回。
  其余阶段不管什么结果都不用自己排查，把两个压缩包原样发回。

时长：Part 1 约 1.5–2.5 小时；两次启动各 20–30 分钟；GPQA 5 个 seed 估计 4–8 小时。合计一个晚上
到一个半晚上，放在 tmux / screen 里或者用 nohup 跑。

工作区：P1 失败时的还原和 P3 写进的分块都只改工作区，下次 git reset --hard 同步代码时会清掉。

用法：在 sglang-jax 仓库根目录下运行。
  python scripts/runbook_flashnext_v7x_stage2.py --model-path /data/Qwen3.8-Flash-Next
  python scripts/runbook_flashnext_v7x_stage2.py ... --stages S3 --gpqa-seeds 59,97   # 断点重跑
  python scripts/runbook_flashnext_v7x_stage2.py ... --dry-run    # 只打印两次启动和调参的命令
中途被打断后重跑之前，先确认没有残留的服务进程占着 TPU（pkill -f sgl_jax.launch_server）。

跑完把 <out>.tar.gz 和 <out>-profile.tar.gz 发回来（后者是 profile trace，体积较大）。
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import os
import re
import shlex
import signal
import statistics
import subprocess
import sys
import tarfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import runbook_flashnext_v7x as rb  # noqa: E402
import runbook_flashnext_v7x_followup as fu  # noqa: E402

STAGES = ("E0", "P1", "P2", "P3", "S1", "S2", "S3", "S4", "S5")
STAGE_INFO = {
    "E0": "环境检查：TPU 设备、权重文件、仓库和代码版本、主机内存",
    "P1": "QSA 稀疏 kernel：真机 parity 测试 + 每个 query 的耗时",
    "P2": "MoE 调参：五个 token 档位的分块",
    "P3": "调好的分块做数值检查，通过的临时写进工作区",
    "S1": "PLE 打开启动：加载摘要、n-gram 表、调好的分块、出 token",
    "S2": "MMLU 冒烟（PLE 开，max_tokens 32768）",
    "S3": "GPQA-Diamond × 5 个 seed（PLE 开，max_tokens 65536）",
    "S4": "压测 512 / 8192 输入（PLE 开）+ profile",
    "S5": "PLE 关闭启动：压测 512 / 8192 输入 + MMLU 冒烟（max_tokens 30720）",
}
rb.STAGE_INFO.update(STAGE_INFO)  # stage_launch 打印阶段标题时查这张表

MIN_MEM_AVAILABLE_GIB = 128

# ---- Part 1 -------------------------------------------------------------------

QSA_PARITY_TEST = "test/srt/kernels/qsa/test_sparse_gqa_parity.py"
QSA_KERNEL = "python/sgl_jax/srt/kernels/qsa/sparse_gqa_attention.py"
QSA_KERNEL_FALLBACK = "4c85cce7"  # 跳过补齐块之前的 sparse_gqa_attention.py
# 模型在 tp=8 下每个设备的形状
QSA_SHAPE = {
    "q_heads": 3,
    "head_dim": 256,
    "page_size": 64,
    "k_blocks": 512,
    "block_units": 128,
    "ratio": 4,
}
QSA_VISIBLE_BLOCKS = (16, 128, 512)
QSA_TOKENS = (16, 2048)  # decode 一批 16 个 query；prefill 一个 chunk 2048 个
QSA_BENCH_ITERS = 20
QSA_TIMEOUT = 1800  # Mosaic 第一次真编译 kernel，编译或 DMA 卡住时不能拖住后面的阶段
QSA_BASELINE_US = {16: 22.7, 2048: 23.5}  # 之前的 profile：每个 query 的耗时，和上下文长度无关

MOE_SHAPE = {"num_experts": 512, "top_k": 10, "hidden_size": 2560, "intermediate_size": 640}
MOE_TOKENS = (16, 64, 512, 1024, 2048)
MOE_TUNE_ARGS = [
    "-m", "benchmark.moe.bench_fused_moe", "--tune-block-config",
    "--num-experts", "512", "--top-k", "10", "--hidden-size", "2560", "--intermediate-size", "640",
    "--num-tokens", *map(str, MOE_TOKENS),
    "--bf-candidates", "128", "640",
    "--bd-candidates", "256", "512", "640", "1280", "2560",
    "--max-configs", "20",
]  # fmt: skip
MOE_TUNE_TIMEOUT = 4 * 3600
MOE_TUNED_LINE = re.compile(
    r"tuned_table\[(?P<device>[^\]]+)\]\[(?P<key>\(.*?\))\] = (?P<value>\(.*?\))"
)
# 调优表的值是 (bt, bf, bd1, bd2, bts, btc, bfc, bd1c, bd2c, bse)；_test_moe 不收 bts，另外传
MOE_BLOCK_NAMES = ("bt", "bf", "bd1", "bd2", None, "btc", "bfc", "bd1c", "bd2c", "bse")
MOE_BTS_INDEX = 4
MOE_CHECK_TIMEOUT = 1800
TUNED_CONFIGS_FILE = "python/sgl_jax/srt/kernels/fused_moe/v1/tuned_block_configs.py"

# ---- Part 2 -------------------------------------------------------------------

LAUNCH_PLE_ON = {
    "ple": True,
    # GPQA 的输出上限 64k，再给题目留 4096（最长的一道题套上 chat template 是 2841 token）
    "context_length": 65536 + 4096,
    "max_running_requests": 64,
    "precompile_bs_paddings": (16, 64),
    "precompile_token_paddings": (16, 64, 512, 1024),
}
LAUNCH_PLE_OFF = dict(rb.DEFAULT_LAUNCH)  # 和第一轮跑通测试相同
EXPECTED_LOAD_SUMMARY_PLE = "consumed=1163, skipped=495, missing=0, unexpected=0"
NGRAM_TABLE_LOADED = "N-gram table: 128/128 shards"
NGRAM_VERIFY_SKIPPED = "keeping the derived values"
TUNED_CONFIG_USED = "Using tuned block config"
READY_TIMEOUT_MIN = 1800  # PLE 打开的那次启动还要读 95 GiB 的 n-gram 表，就绪等待至少给这么久

MMLU_URL = "https://openaipublic.blob.core.windows.net/simple-evals/mmlu.csv"
GPQA_URL = "https://openaipublic.blob.core.windows.net/simple-evals/gpqa_diamond.csv"
# 和 test/srt/test_qwen3_5_models.py 的 _run_mmlu_smoke 完全一致
MMLU_JOB = {
    "eval": "mmlu", "num_examples": 100, "threads": 16, "seed": 17,
    "sampling": {
        "temperature": 0.6, "top_p": 0.95, "top_k": 20, "min_p": 0.0,
        "presence_penalty": 1.5, "repetition_penalty": 1.0, "max_tokens": 32768,
    },
}  # fmt: skip
MMLU_PASS, MMLU_FLOOR = 0.70, 0.30
# PLE 关闭那次上下文是 32768，输入加输出达到它的请求会被拒绝
MMLU_PLE_OFF_MAX_TOKENS = 32768 - 2048
# 模型卡推荐的 thinking 采样参数
GPQA_JOB = {
    "eval": "gpqa", "num_examples": None, "threads": 64,
    "sampling": {
        "temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0.0,
        "presence_penalty": 0.0, "repetition_penalty": 1.0, "max_tokens": 65536,
    },
}  # fmt: skip
GPQA_SEEDS = "17,23,41,59,97"
GPQA_PUBLISHED = 0.917
# 第一轮 PLE 关闭时 qsa_sparse 的压测（10-02 那次跑通测试），S4 / S5 拿来对照
PERF_BASELINE = {
    "512": {"completed": 100, "median_ttft_ms": 2606.6, "median_tpot_ms": 74.5,
            "output_throughput": 83.0, "total_throughput": 415.2},
    "8192": {"completed": 32, "median_ttft_ms": 31471.4, "median_tpot_ms": 222.2,
             "output_throughput": 17.2, "total_throughput": 1117.2},
}  # fmt: skip
EVAL_RETRIES = 5  # 400 和超时以外的错误最多重试这么多次，不无限退避
# 一个 64k token 的回答在 bs 64 下要一个多小时；超时就记下来，不重发（重发等于从头再生成一遍）
EVAL_REQUEST_TIMEOUT = 6 * 3600

PROFILE_DECODE = {**fu.PROFILE_REQUEST, "profile_stages": ["decode"]}

RANK = {"PASS": 0, "WARN": 1, "FAIL": 2}


# ---- 子进程 -------------------------------------------------------------------


def run_worker(run, args, kind: str, spec: dict, name: str, srv=None, timeout=None) -> dict:
    """在子进程里跑本脚本的一个 worker，返回它写出的 JSON；失败时返回 {"error": ...}。"""
    result = run.out / f"{name}.json"
    result.unlink(missing_ok=True)
    spec = {
        **spec,
        "repo": args.repo,
        "out": str(run.out.resolve()),
        "result": str(result.resolve()),
    }
    cmd = [sys.executable, str(Path(__file__).resolve()), "--_worker", kind, json.dumps(spec)]
    rc = rb.run_logged(args, cmd, run.out / f"{name}.log", srv=srv, timeout=timeout)
    if rc != 0 or not result.exists():
        return {"error": f"返回码 {rc}，请查看 {name}.log"}
    return json.loads(result.read_text())


def worker_qsa_bench(spec: dict) -> dict:
    import jax
    import jax.numpy as jnp
    import numpy as np

    from sgl_jax.srt.kernels.qsa.sparse_gqa_attention import sparse_gqa_attention

    sh = QSA_SHAPE
    rng = np.random.default_rng(0)
    # 每个 query 前面都有 512 个以上已封口的块，可见块数由 block_ids 里有效 id 的个数决定
    ctx = sh["k_blocks"] * sh["ratio"] + sh["page_size"]
    pages_per_seq = -(-ctx // sh["page_size"])
    n_reqs = 4
    n_pages = n_reqs * pages_per_seq
    kv = rng.standard_normal((n_pages, sh["page_size"], 1, 2, sh["head_dim"])).astype(np.float32)
    cache = jnp.asarray(kv, jnp.bfloat16)  # bf16 打包布局：K、V 共用一个 32 位字
    page_table = jnp.asarray(rng.permutation(n_pages).reshape(n_reqs, pages_per_seq), jnp.int32)
    rows = []
    for t in QSA_TOKENS:
        q = jnp.asarray(
            rng.standard_normal((t, sh["q_heads"], sh["head_dim"])).astype(np.float32),
            jnp.bfloat16,
        )
        pos = jnp.full((t,), ctx - 1, jnp.int32)
        req = jnp.asarray(rng.integers(0, n_reqs, t), jnp.int32)
        for visible in QSA_VISIBLE_BLOCKS:
            blk = np.full((t, sh["k_blocks"]), -1, np.int32)
            for i in range(t):
                blk[i, :visible] = rng.permutation(sh["k_blocks"])[:visible]
            blk = jnp.asarray(blk)

            def call():
                return sparse_gqa_attention(
                    q, blk, pos, req, page_table, cache,
                    sm_scale=sh["head_dim"] ** -0.5, ratio=sh["ratio"],
                    block_units=sh["block_units"],
                )  # fmt: skip

            call().block_until_ready()  # 编译
            # 连发再同步一次：16 个 query 的设备时间只有几百 µs，逐次同步会把 dispatch 摊进去
            t0 = time.perf_counter()
            jax.block_until_ready([call() for _ in range(QSA_BENCH_ITERS)])
            per_call = (time.perf_counter() - t0) / QSA_BENCH_ITERS
            rows.append(
                {
                    "tokens": t,
                    "visible_blocks": visible,
                    "ms_per_call": round(per_call * 1e3, 3),
                    "us_per_query": round(per_call / t * 1e6, 2),
                }
            )
    return {"rows": rows}


def worker_moe_check(spec: dict) -> dict:
    import functools

    import jax.numpy as jnp

    import sgl_jax.test.kernels.fused_moe_v1_test as moe_test

    # _test_moe 构造分块配置时不传 bts，kernel 就用 bts=bt；服务端用的是调好的 bts。
    # 让测试里那个名字带上调好的 bts，检查的才是写进调优表的那一组分块。
    moe_test.FusedMoEBlockConfig = functools.partial(moe_test.FusedMoEBlockConfig, bts=spec["bts"])
    case = moe_test.MoEKernelTest()
    case.setUp()
    try:
        # 和 fused_moe_v1_test 的 test_flash_next_fallback_tiles 同一个调用，只是专家数和分块不同
        case._test_moe(
            dtype=jnp.bfloat16,
            top_k=MOE_SHAPE["top_k"],
            num_experts=MOE_SHAPE["num_experts"],
            hidden_size=MOE_SHAPE["hidden_size"],
            intermediate_size=MOE_SHAPE["intermediate_size"],
            num_tokens=spec["num_tokens"],
            seed=54321,
            renormalize_topk_logits=True,
            act_fn="silu",
            atol=5e-2,
            rtol=5e-2,
            **spec["blocks"],
        )
    finally:
        case.tearDown()
    return {"passed": True}


def worker_eval(spec: dict) -> dict:
    """跑一次 MMLU / GPQA，记下每个请求的 finish_reason、输出 token 数和抽到的答案。"""
    import threading

    sys.path.insert(0, str(Path(spec["repo"], "test", "srt")))
    import openai
    from eval.simple_eval_common import (
        ANSWER_PATTERN_MULTICHOICE,
        ChatCompletionSampler,
        make_report,
        set_ulimit,
        strip_reasoning,
    )

    # 和 test/srt/run_eval.py 一样：本地服务不校验 key，OpenAI 客户端却要求有一个
    os.environ.setdefault("OPENAI_API_KEY", "EMPTY")
    set_ulimit()

    class RecordingSampler(ChatCompletionSampler):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.client = self.client.with_options(timeout=EVAL_REQUEST_TIMEOUT, max_retries=0)
            self.calls, self.lock = [], threading.Lock()

        def __call__(self, message_list):
            kwargs = dict(
                model=self.model,
                messages=message_list,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            if self.top_p is not None:
                kwargs["top_p"] = self.top_p
            if self.extra_body:
                kwargs["extra_body"] = self.extra_body
            text, record = "", {}
            for attempt in range(EVAL_RETRIES):
                try:
                    resp = self.client.chat.completions.create(**kwargs)
                except openai.BadRequestError as e:
                    record = {"error": f"BadRequestError: {e}", "rejected": True}
                    break
                except openai.APITimeoutError as e:
                    record = {"error": f"APITimeoutError: {e}", "timeout": True}
                    break
                except Exception as e:  # noqa: BLE001 - 有上限地重试
                    record = {"error": f"{type(e).__name__}: {e}"}
                    time.sleep(min(2**attempt, 60))
                    continue
                choice = resp.choices[0]
                text = choice.message.content or ""
                reasoning = getattr(choice.message, "reasoning_content", None)
                if reasoning:
                    text = f"{reasoning}\n{text}"
                match = re.search(ANSWER_PATTERN_MULTICHOICE, strip_reasoning(text))
                record = {
                    "finish_reason": choice.finish_reason,
                    "completion_tokens": resp.usage.completion_tokens if resp.usage else None,
                    "pred": match.group(1) if match else None,
                }
                break
            with self.lock:
                self.calls.append(record)
            return text

    if spec["eval"] == "mmlu":
        from eval.simple_eval_mmlu import MMLUEval

        eval_obj = MMLUEval(MMLU_URL, spec["num_examples"], spec["threads"])
    else:
        from eval.simple_eval_gpqa import GPQAEval

        eval_obj = GPQAEval(GPQA_URL, spec["num_examples"], spec["threads"])
    sampling = dict(spec["sampling"])
    extra = {
        k: sampling.pop(k) for k in ("top_k", "min_p", "presence_penalty", "repetition_penalty")
    }
    extra.update(seed=spec["seed"], chat_template_kwargs={"enable_thinking": True})
    sampler = RecordingSampler(
        base_url=f"{spec['base_url']}/v1",
        model=spec["model"],
        temperature=sampling["temperature"],
        max_tokens=sampling["max_tokens"],
        top_p=sampling["top_p"],
        extra_body=extra,
    )
    result = eval_obj(sampler)
    stem = Path(spec["result"]).with_suffix("")
    Path(f"{stem}.html").write_text(make_report(result))
    with open(f"{stem}.calls.jsonl", "w") as f:
        for record in sampler.calls:
            f.write(json.dumps(record) + "\n")
    metrics = {k: float(v) for k, v in result.metrics.items() if isinstance(v, (int, float))}
    return {"score": float(result.score), "metrics": metrics, **summarize_calls(sampler.calls)}


def summarize_calls(calls: list[dict]) -> dict:
    tokens = sorted(c["completion_tokens"] for c in calls if c.get("completion_tokens") is not None)

    def pct(p):
        return tokens[min(len(tokens) - 1, int(p * len(tokens)))] if tokens else None

    return {
        "requests": len(calls),
        "finish_length": sum(c.get("finish_reason") == "length" for c in calls),
        "pred_none": sum("error" not in c and c.get("pred") is None for c in calls),
        "rejected": sum(bool(c.get("rejected")) for c in calls),
        "timeouts": sum(bool(c.get("timeout")) for c in calls),
        "errors": sum("error" in c for c in calls),
        "output_tokens": {"min": pct(0), "p50": pct(0.5), "p90": pct(0.9), "max": pct(1)},
    }


WORKERS = {"qsa-bench": worker_qsa_bench, "moe-check": worker_moe_check, "eval": worker_eval}


# ---- Part 1 -------------------------------------------------------------------


def stage_p1(run: rb.Run, args) -> None:
    run.log(f"===== P1：{STAGE_INFO['P1']}")
    rb.git(args.repo, "checkout", "HEAD", "--", QSA_KERNEL)  # 从仓库里的版本开始测
    parity = [sys.executable, QSA_PARITY_TEST]
    rc = rb.run_logged(args, parity, run.out / "p1_qsa_parity.log", timeout=QSA_TIMEOUT)
    run.log(f"  parity 测试：返回码 {rc}")
    bench = run_worker(run, args, "qsa-bench", {}, "p1_qsa_bench", timeout=QSA_TIMEOUT)
    for row in bench.get("rows", []):
        run.log(f"  {row}")
    details = {"parity_returncode": rc, "bench": bench, "baseline_us": QSA_BASELINE_US}
    if rc == 0 and "error" not in bench:
        run.record("P1", "PASS", **details)
        return
    reason = (
        "parity 测试超时"
        if rc == -9
        else "parity 测试没过" if rc != 0 else f"基准没跑成：{bench['error']}"
    )
    restored = rb.git(args.repo, "checkout", QSA_KERNEL_FALLBACK, "--", QSA_KERNEL).returncode == 0
    if restored:
        rc_fallback = rb.run_logged(
            args, parity, run.out / "p1_qsa_parity_fallback.log", timeout=QSA_TIMEOUT
        )
        after = f"已把工作区的 sparse_gqa_attention.py 还原成 {QSA_KERNEL_FALLBACK} 的版本，" + (
            "还原后 parity 测试通过"
            if rc_fallback == 0
            else f"还原后 parity 测试也没过（返回码 {rc_fallback}）"
        )
    else:
        rc_fallback = None
        after = f"还原成 {QSA_KERNEL_FALLBACK} 的版本失败，Part 2 用的仍是仓库里的 kernel"
    run.log(f"  {reason}；{after}")
    run.record(
        "P1",
        "FAIL",
        error=f"{reason}；{after}",
        restored=restored,
        fallback_parity_returncode=rc_fallback,
        **details,
    )


def stage_p2(run: rb.Run, args) -> None:
    run.log(f"===== P2：{STAGE_INFO['P2']}")
    log = run.out / "p2_moe_tune.log"
    rc = rb.run_logged(args, [sys.executable, *MOE_TUNE_ARGS], log, timeout=MOE_TUNE_TIMEOUT)
    parsed = [
        {
            "device": m["device"],
            "key": list(ast.literal_eval(m["key"])),
            "value": list(ast.literal_eval(m["value"])),
        }
        for m in MOE_TUNED_LINE.finditer(log.read_text(errors="replace"))
    ]
    # 同一个 key 出现多次时，以最后一次为准
    entries = list({(e["device"], tuple(e["key"])): e for e in parsed}.values()) if rc == 0 else []
    (run.out / "moe_tuned.json").write_text(json.dumps({"entries": entries}, indent=2))
    run.log(f"  调参：返回码 {rc}，解析出 {len(parsed)} 条，保留 {len(entries)} 条")
    missing = sorted(set(MOE_TOKENS) - {e["key"][2] for e in entries})
    if rc == 0 and entries:
        if missing:
            run.log(f"  这些 token 数没有调出条目，用回退分块：{missing}")
        run.record("P2", "WARN" if missing else "PASS", entries=entries, missing_tokens=missing)
    else:
        why = "超时" if rc == -9 else f"返回码 {rc}" if rc != 0 else "没有输出条目"
        run.record(
            "P2", "FAIL", error=f"调参{why}，不写任何条目，Part 2 用回退分块", parsed=len(parsed)
        )


def write_tuned_entries(repo: str, entries: list[dict]) -> None:
    """把条目追加到对应设备段的末尾：字典字面量里后出现的同名 key 覆盖前面的。"""
    path = Path(repo, TUNED_CONFIGS_FILE)
    lines = path.read_text().splitlines(keepends=True)
    for device in dict.fromkeys(e["device"] for e in entries):
        head = f'    "{device}": {{\n'
        if head not in lines:
            raise RuntimeError(f"{TUNED_CONFIGS_FILE} 里没有设备段 {device!r}")
        start = lines.index(head)
        end = next(i for i in range(start + 1, len(lines)) if lines[i] == "    },\n")
        new = [
            f"        {tuple(e['key'])!r}: {tuple(e['value'])!r},\n"
            for e in entries
            if e["device"] == device
        ]
        lines[end:end] = [
            "        # runbook_flashnext_v7x_stage2.py 调参并通过数值检查的条目\n",
            *new,
        ]
    path.write_text("".join(lines))


def tuned_entries_in_table(args, entries: list[dict]) -> bool:
    """在子进程里导入调优表，确认文件还能解析、条目都在。"""
    probe = (
        "import json, sys\n"
        "from sgl_jax.srt.kernels.fused_moe.v1.tuned_block_configs import TUNED_BLOCK_CONFIGS as T\n"
        "es = json.loads(sys.argv[1])\n"
        "ok = all(list(T[e['device']].get(tuple(e['key']), ())) == e['value'] for e in es)\n"
        "sys.exit(0 if ok else 1)\n"
    )
    r = subprocess.run(
        [sys.executable, "-c", probe, json.dumps(entries)], cwd=args.repo, check=False
    )
    return r.returncode == 0


def stage_p3(run: rb.Run, args) -> None:
    run.log(f"===== P3：{STAGE_INFO['P3']}")
    rb.git(args.repo, "checkout", "HEAD", "--", TUNED_CONFIGS_FILE)  # 从仓库里的版本开始写
    tuned = run.out / "moe_tuned.json"
    entries = json.loads(tuned.read_text())["entries"] if tuned.exists() else []
    if not entries:
        run.record("P3", rb.NOT_RUN, reason="P2 没有调出条目，Part 2 用回退分块")
        return
    checks, passed = [], []
    for e in entries:
        blocks = {name: v for name, v in zip(MOE_BLOCK_NAMES, e["value"]) if name}
        n = e["key"][2]  # key 的第 3 项是 token 数
        check = {"num_tokens": n, "blocks": blocks, "bts": e["value"][MOE_BTS_INDEX]}
        res = run_worker(
            run, args, "moe-check", check,
            f"p3_moe_check_{n}", timeout=MOE_CHECK_TIMEOUT,
        )  # fmt: skip
        ok = res.get("passed", False)
        run.log(f"  token 数 {n}：{'通过' if ok else '没通过'}  {e['value']}")
        checks.append({"num_tokens": n, "value": e["value"], "passed": ok, **res})
        if ok:
            passed.append(e)
    if not passed:
        run.record("P3", "FAIL", error="没有条目通过数值检查，Part 2 用回退分块", checks=checks)
        return
    try:
        write_tuned_entries(args.repo, passed)
        if not tuned_entries_in_table(args, passed):
            raise RuntimeError("写完之后导入调优表，条目对不上")
    except Exception as e:  # noqa: BLE001 - 写坏了就还原，Part 2 用回退分块
        rb.git(args.repo, "checkout", "HEAD", "--", TUNED_CONFIGS_FILE)
        run.record("P3", "FAIL", error=f"写进调优表失败，已还原：{e}", checks=checks)
        return
    diff = rb.git(args.repo, "diff", "--", TUNED_CONFIGS_FILE).stdout
    (run.out / "p3_tuned_configs.diff").write_text(diff)
    status = "PASS" if len(passed) == len(entries) else "WARN"
    run.record("P3", status, checks=checks, written=passed)


# ---- Part 2 -------------------------------------------------------------------


def run_eval_job(run, srv, args, name: str, job: dict, seed: int) -> dict:
    spec = {**job, "seed": seed, "base_url": srv.base, "model": args.model_path}
    run.log(f"  {job['eval']}（{srv.label}，seed {seed}）：{job['num_examples'] or '全部'} 道题")
    res = run_worker(run, args, "eval", spec, name, srv=srv)
    if reason := srv.dead_reason():  # 评测刚结束服务就退出了：最后一批请求的结果不可信
        raise RuntimeError(reason)
    run.log(f"  {job['eval']} 得分：{res.get('score', res.get('error'))}")
    if res.get("errors"):
        run.log(
            f"  有 {res['errors']} 个请求出错（被拒 {res.get('rejected', 0)}，超时 {res.get('timeouts', 0)}）"
        )
    return res


def mmlu_status(run, res: dict) -> str:
    if "score" not in res:
        return "FAIL"
    if res["score"] > MMLU_PASS:
        return "WARN" if res.get("errors") else "PASS"
    if res["score"] < MMLU_FLOOR:
        run.log(
            f"  MMLU 低于 {MMLU_FLOOR}，接近随机水平：先查 {res.get('rejected', 0)} 个被拒的请求"
            "和服务日志"
        )
        return "WARN"
    return "FAIL"


def measure_perf(run, srv, args, profile: dict | None) -> tuple[dict, list[str]]:
    bench, errors = {}, []
    for cfg in fu.BENCH_SWEEP:
        key = str(cfg["input_len"])
        try:
            bench[key] = rb.run_bench(run, srv, args, cfg, tag=f"_in{key}")
            run.log(f"  bench_serving（{srv.label}，输入 {key}）：{bench[key]}")
        except Exception as e:  # noqa: BLE001 - 一组失败，另一组照样跑
            errors.append(f"bench_serving（输入 {key}）：{type(e).__name__}: {e}")
    result = {"bench": bench}
    if profile is not None:
        try:
            result["profile"] = fu.capture_profile(run, srv, args, request=profile)
            run.log(f"  profile（{srv.label}）：{result['profile']}")
        except Exception as e:  # noqa: BLE001
            errors.append(f"profile：{type(e).__name__}: {e}")
    for e in errors:
        run.log(f"  性能测量出错：{e}")
    return {**result, "perf_errors": errors}, errors


def stage_ple_on(run: rb.Run, args, wanted: set[str]) -> None:
    def s1(srv, ready, checks):
        log = srv.log_path.read_text(errors="replace")
        problems, warnings = [], []
        if not checks["load_summary"] or EXPECTED_LOAD_SUMMARY_PLE not in checks["load_summary"]:
            problems.append(f"加载摘要不等于 {EXPECTED_LOAD_SUMMARY_PLE}")
        if NGRAM_TABLE_LOADED not in log:
            problems.append(f"日志里没有 {NGRAM_TABLE_LOADED!r}，n-gram 表没有装完")
        skipped = [line for line in log.splitlines() if NGRAM_VERIFY_SKIPPED in line]
        if skipped:
            warnings.append(f"哈希校验有 {len(skipped)} 项被跳过：{skipped[0].strip()}")
        written = run.results.get("P3", {}).get("written", [])
        if written and TUNED_CONFIG_USED not in log:
            warnings.append(f"P3 写进了 {len(written)} 条分块，但日志里没有 {TUNED_CONFIG_USED!r}")
        smoke = srv.generate(rb.PROMPTS[0], 16)
        if not smoke["text"].strip() or any(math.isnan(x) for x in smoke["logprobs"]):
            problems.append("冒烟请求输出为空或 logprob 里有 NaN")
        run.log(f"  冒烟请求：{rb.PROMPTS[0]!r} -> {smoke['text']!r}")
        for p in problems + warnings:
            run.log(f"  问题：{p}")
        status = "FAIL" if problems else "WARN" if warnings else "PASS"
        run.record(
            "S1",
            status,
            problems=problems + warnings,
            ready_minutes=round(ready / 60, 1),
            smoke=smoke["text"],
            tuned_config_used=TUNED_CONFIG_USED in log,
            **{
                **checks,
                "load_summary_ok": EXPECTED_LOAD_SUMMARY_PLE in (checks["load_summary"] or ""),
            },
        )

    def s2(srv, ready, checks):
        res = run_eval_job(run, srv, args, "s2_mmlu", MMLU_JOB, MMLU_JOB["seed"])
        run.record("S2", mmlu_status(run, res), mmlu=res)

    def s3(srv, ready, checks):
        s1_status = run.results.get("S1", {}).get("status")
        s2_score = run.results.get("S2", {}).get("mmlu", {}).get("score")
        if s1_status == "FAIL" or (s2_score is not None and s2_score < MMLU_FLOOR):
            reason = "S1 没有通过" if s1_status == "FAIL" else f"S2 的 MMLU 只有 {s2_score}"
            run.record("S3", rb.NOT_RUN, reason=f"{reason}，模型已经不对，GPQA 没有意义")
            return
        # 只重跑 GPQA 时（这次没有 S1）保留没重跑的 seed；整轮重跑时从头来
        runs = {} if "S1" in wanted else dict(run.results.get("S3", {}).get("runs", {}))
        try:
            for seed in [int(s) for s in args.gpqa_seeds.split(",") if s.strip()]:
                job = run_eval_job(run, srv, args, f"s3_gpqa_seed{seed}", GPQA_JOB, seed)
                runs[str(seed)] = job
                run.results["S3"] = {"status": "未完成", "runs": runs}
                run.save()
        except Exception as e:  # noqa: BLE001 - 服务退出时，已经跑完的 seed 照样留下
            run.record("S3", "FAIL", error=f"{type(e).__name__}: {e}", runs=runs)
            return
        scores = [r["score"] for r in runs.values() if "score" in r]
        summary = {"runs": runs, "published": GPQA_PUBLISHED}
        if scores:
            med = statistics.median(scores)
            summary.update(median=round(med, 4), delta=round(med - GPQA_PUBLISHED, 4))
            run.log(f"  GPQA 中位数 {med:.4f}（{len(scores)} 个 seed），模型卡 {GPQA_PUBLISHED}")
        clean = (
            scores and len(scores) == len(runs) and not any(r.get("errors") for r in runs.values())
        )
        run.record("S3", "PASS" if clean else "WARN" if scores else "FAIL", **summary)

    def s4(srv, ready, checks):
        profile = fu.PROFILE_REQUEST if args.profile_prefill else PROFILE_DECODE
        perf, errors = measure_perf(run, srv, args, profile)
        run.record("S4", "FAIL" if not perf["bench"] else "WARN" if errors else "PASS", **perf)

    steps = [(s, fn) for s, fn in (("S1", s1), ("S2", s2), ("S3", s3), ("S4", s4)) if s in wanted]
    rb.stage_launch(run, args, "qsa_sparse", steps, LAUNCH_PLE_ON, "ple_on")


def stage_ple_off(run: rb.Run, args) -> None:
    def s5(srv, ready, checks):
        perf, errors = measure_perf(run, srv, args, None)
        job = {
            **MMLU_JOB,
            "sampling": {**MMLU_JOB["sampling"], "max_tokens": MMLU_PLE_OFF_MAX_TOKENS},
        }
        try:
            mmlu = run_eval_job(run, srv, args, "s5_mmlu", job, job["seed"])
        except Exception as e:  # noqa: BLE001 - 服务退出时，压测数字照样留下
            mmlu = {"error": f"{type(e).__name__}: {e}"}
        if "score" not in mmlu:
            errors.append(f"MMLU：{mmlu.get('error')}")
        elif mmlu.get("errors"):
            errors.append(f"MMLU：{mmlu['errors']} 个请求出错")
        nothing = not perf["bench"] and "score" not in mmlu
        run.record(
            "S5",
            "FAIL" if nothing else "WARN" if errors else "PASS",
            **perf,
            mmlu=mmlu,
            load_summary=checks["load_summary"],
        )

    rb.stage_launch(run, args, "qsa_sparse", [("S5", s5)], LAUNCH_PLE_OFF, "ple_off")


# ---- 报告 ---------------------------------------------------------------------


def fmt(x) -> str:
    return f"{x:.1f}" if isinstance(x, float) else str(x)


def write_report(run: rb.Run, repo: dict) -> Path:
    r = run.results
    lines = [
        "# Qwen3.8-Flash-Next 第二轮测试：kernel 验证 + MoE 调参 + PLE 打开",
        "",
        f"代码：{repo.get('branch')} @ {repo.get('head')}"
        + ("（工作区有改动，见 P1 / P3）" if repo.get("dirty") else ""),
        "",
        "| 阶段 | 结果 | 测什么 |",
        "| --- | --- | --- |",
    ]
    lines += [
        f"| {s} | {r.get(s, {}).get('status', rb.NOT_RUN)} | {STAGE_INFO[s]} |" for s in STAGES
    ]
    lines += [""]
    for s in STAGES:
        c = r.get(s, {})
        lines += [f"- {s} 问题：{p}" for p in c.get("problems", [])]
        lines += [f"- {s}：{c[k]}" for k in ("error", "reason") if k in c]
        lines += [f"- {s} 性能测量出错：{e}" for e in c.get("perf_errors", [])]

    rows = r.get("P1", {}).get("bench", {}).get("rows", [])
    if rows:
        lines += [
            "",
            f"## P1 稀疏 kernel 每个 query 的耗时（连发 {QSA_BENCH_ITERS} 次再同步一次，取平均）",
            "",
            "| query 数 | 可见块数 | 每个 query µs | 之前 profile µs |",
            "| --- | --- | --- | --- |",
        ]
        lines += [
            f"| {x['tokens']} | {x['visible_blocks']} | {x['us_per_query']} | "
            f"{QSA_BASELINE_US.get(x['tokens'], '-')} |"
            for x in rows
        ]
    p3 = r.get("P3", {}).get("checks", [])
    if p3:
        lines += [
            "",
            "## P3 调好的分块",
            "",
            "| token 数 | 分块 | 数值检查 |",
            "| --- | --- | --- |",
        ]
        lines += [
            f"| {c['num_tokens']} | {tuple(c['value'])} | {'通过' if c['passed'] else '没通过'} |"
            for c in p3
        ]
        diff = run.out / "p3_tuned_configs.diff"
        if diff.exists():
            lines += ["", "写进工作区的改动：", "", "```diff", diff.read_text().rstrip(), "```"]
    lines += ["", "工作区相对 HEAD 的全部改动在 workspace.diff。"]
    lines += ["", "CPU 单测：在本地跑，不在这次运行里。"]
    s1 = r.get("S1", {})
    if "load_summary" in s1:
        lines += ["", f"S1 加载摘要：`{s1['load_summary']}`（期望 `{EXPECTED_LOAD_SUMMARY_PLE}`）"]

    evals = [("S2 MMLU（PLE 开，max_tokens 32768）", r.get("S2", {}).get("mmlu"))]
    evals += [
        (f"S3 GPQA seed {seed}", res) for seed, res in r.get("S3", {}).get("runs", {}).items()
    ]
    evals += [
        (f"S5 MMLU（PLE 关，max_tokens {MMLU_PLE_OFF_MAX_TOKENS}）", r.get("S5", {}).get("mmlu"))
    ]
    evals = [(n, e) for n, e in evals if e]
    if evals:
        lines += [
            "",
            "## 精度",
            "",
            "| 评测 | 分数 | 请求数 | 截断 | 没抽到答案 | 被拒 | 超时 | 出错 | 输出 token p50 / p90 / max |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
        for name, e in evals:
            if "score" not in e:
                lines += [f"| {name} | 出错：{e.get('error')} | | | | | | | |"]
                continue
            t = e.get("output_tokens") or {}
            names = ("requests", "finish_length", "pred_none", "rejected", "timeouts", "errors")
            counts = [e.get(k, "-") for k in names]
            tokens = " / ".join(str(t.get(k, "-")) for k in ("p50", "p90", "max"))
            lines += [
                f"| {name} | {e['score']:.4f} | " + " | ".join(map(str, counts)) + f" | {tokens} |"
            ]
        s3 = r.get("S3", {})
        if "median" in s3:
            lines += [
                "",
                f"GPQA-Diamond 中位数 {s3['median']}，模型卡 {GPQA_PUBLISHED}，差 {s3['delta']:+.4f}。"
                "seed 只用来区分 5 次运行：服务端目前不使用请求里的 seed，重跑同一个 seed 不会复现。",
            ]
        s2, s5 = r.get("S2", {}).get("mmlu", {}), r.get("S5", {}).get("mmlu", {})
        if "score" in s2 and "score" in s5:
            lines += [
                f"MMLU PLE 开 − 关：{s2['score'] - s5['score']:+.4f}"
                f"（两边 max_tokens 不同：32768 对 {MMLU_PLE_OFF_MAX_TOKENS}）"
            ]

    perf = [
        (label, key, m)
        for s, label in (("S4", "PLE 开"), ("S5", "PLE 关"))
        for key, m in r.get(s, {}).get("bench", {}).items()
    ]
    if perf:
        perf += [("10-02 第一轮（PLE 关）", key, m) for key, m in PERF_BASELINE.items()]
    if perf:
        lines += [
            "",
            "## 性能（qsa_sparse）",
            "",
            "| PLE | 输入长度 | 完成数 | TTFT 中位数 ms | TPOT 中位数 ms | 输出吞吐 tok/s | 总吞吐 tok/s |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
        keys = (
            "completed",
            "median_ttft_ms",
            "median_tpot_ms",
            "output_throughput",
            "total_throughput",
        )
        lines += [
            f"| {label} | {key} | " + " | ".join(fmt(m.get(k, "-")) for k in keys) + " |"
            for label, key, m in perf
        ]
    if r.get("S4", {}).get("profile"):
        lines += ["", f"S4 profile：{r['S4']['profile']}"]
    path = run.out / "report.md"
    path.write_text("\n".join(lines) + "\n")
    return path


def pack(run: rb.Run, archive: str, keep) -> None:
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(run.out, arcname=run.out.name, filter=lambda t: t if keep(t.name) else None)


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] == "--_worker":
        spec = json.loads(sys.argv[3])
        Path(spec["result"]).write_text(json.dumps(WORKERS[sys.argv[2]](spec)))
        return 0

    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--model-path", required=True, help="完整 checkpoint 所在的本地目录")
    ap.add_argument("--repo", default=os.getcwd(), help="sglang-jax 仓库根目录（默认当前目录）")
    ap.add_argument("--out", default="bringup-stage2", help="输出目录")
    ap.add_argument("--port", type=int, default=30000, help="服务端口")
    ap.add_argument("--tp", type=int, default=rb.EXPECTED_DEVICES, help="张量并行度（默认 8）")
    ap.add_argument("--start-timeout", type=int, default=7200, help="每次启动最多等多少秒")
    ap.add_argument("--stages", default=",".join(STAGES), help="要跑的阶段，逗号分隔")
    ap.add_argument("--gpqa-seeds", default=GPQA_SEEDS, help="GPQA-Diamond 的 seed，逗号分隔")
    ap.add_argument(
        "--profile-prefill", action="store_true", help="profile 连 prefill 一起抓（体积大）"
    )
    ap.add_argument(
        "--extra-server-args",
        nargs=argparse.REMAINDER,
        default=[],
        help="追加给 launch_server 的参数（放在命令最后）",
    )
    ap.add_argument("--dry-run", action="store_true", help="只打印两次启动和调参的命令")
    args = ap.parse_args()
    args.model_path = str(Path(args.model_path).resolve())
    args.repo = str(Path(args.repo).resolve())
    args.start_timeout = max(args.start_timeout, READY_TIMEOUT_MIN)

    if args.dry_run:
        for name, launch in (("S1-S4（PLE 开）", LAUNCH_PLE_ON), ("S5（PLE 关）", LAUNCH_PLE_OFF)):
            print(f"# {name}\n{shlex.join(rb.server_cmd(args, 'qsa_sparse', launch))}\n")
        print(f"# P1\n{shlex.join([sys.executable, QSA_PARITY_TEST])}\n")
        print(f"# P2\n{shlex.join([sys.executable, *MOE_TUNE_ARGS])}")
        return 0

    # SSH 断开或被 kill 时照 Ctrl-C 处理：先停掉服务，再写报告、打包
    signal.signal(signal.SIGHUP, signal.default_int_handler)
    signal.signal(signal.SIGTERM, signal.default_int_handler)
    run = rb.Run(Path(args.out))
    stages = {s.strip().upper() for s in args.stages.split(",")}
    for s in stages - {"S3"}:  # S3 按 seed 落盘，断点重跑时保留没重跑的 seed
        run.results.pop(s, None)
    run.save()
    run.log(f"要跑的阶段：{sorted(stages)}；输出目录：{run.out.resolve()}")

    e0_failed = False
    try:
        if "E0" in stages and not rb.stage_c0(run, args, "E0", MIN_MEM_AVAILABLE_GIB):
            e0_failed = True
            stages = set()
        for stage, fn in (("P1", stage_p1), ("P2", stage_p2), ("P3", stage_p3)):
            if stage in stages:
                try:
                    fn(run, args)
                except Exception as e:  # noqa: BLE001 - Part 1 的失败不阻断 Part 2
                    run.log(f"{stage} 出错：{type(e).__name__}: {e}")
                    run.record(stage, "FAIL", error=f"{type(e).__name__}: {e}")
        if stages & {"S1", "S2", "S3", "S4"}:
            stage_ple_on(run, args, stages)
        if "S5" in stages:
            stage_ple_off(run, args)
    finally:
        (run.out / "workspace.diff").write_text(rb.git(args.repo, "diff", "HEAD").stdout)
        run.log(f"报告：{write_report(run, rb.git_info(args.repo))}")
        if e0_failed:
            run.log("环境检查（E0）没有通过：按上面列出的问题修好后重跑，这次不打包、不用发回")
        else:
            main_archive, profile_archive = f"{run.out}.tar.gz", f"{run.out}-profile.tar.gz"
            pack(run, main_archive, lambda n: "jit_cache" not in n and "/profile_" not in n)
            pack(run, profile_archive, lambda n: n == run.out.name or "/profile_" in n)
            run.log(f"请把这两个压缩包发回：{main_archive}、{profile_archive}")
    return 0 if all(v["status"] != "FAIL" for v in run.results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
