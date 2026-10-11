"""Qwen3.8-Flash-Next 在 TPU v7x 上的性能追平测试（R1）：同一套启动参数下，测四种组合的精度和
吞吐，并给官方 FP8 checkpoint 调 MoE 分块。

  A  基底 A（当前的模型代码）+ bf16 checkpoint：基线。启动命令和 cookbook §2.3 完全一致，
     顺带验证 cookbook §3 的三个调用示例。
  B  基底 B（预算以内的 QSA 层直接走稠密注意力，并能加载 FP8 checkpoint）+ bf16 checkpoint：
     和 A 的差就是稠密分支的收益。
  C  基底 B + 官方 FP8 checkpoint：先调 FP8 MoE 的分块，再测精度和吞吐。
  D  基底 D（N-gram 表放进 HBM、在设备上查表，overlap 调度打开）+ FP8 checkpoint，和 C 用同一套
     MoE 分块：和 C 的差就是这一步的收益。最后用 bf16 checkpoint 再启动一次，确认表放得下。

和 runbook_flashnext_v7x.py 用同一套环境（前置条件见那份脚本的说明），另外要求：
  * FP8 checkpoint：Hugging Face 上的 Qwen/Qwen3.8-Flash-Next-FP8（约 173 GiB，131 个
    safetensors 文件），事先下好：
      hf download Qwen/Qwen3.8-Flash-Next-FP8 --local-dir /data/Qwen3.8-Flash-Next-FP8
  * 工作区干净：每一部分开始前，脚本把仓库 git checkout 到这一部分的基底（写死的提交，
    都在 bring-up 分支的历史里），全部跑完再切回原来的分支。

各次启动的参数相同（上下文 69632、64 个请求，和 cookbook §2.3 一致），只换代码和
checkpoint；A1 另外带上 cookbook 的两个 parser，Part D 不带 --disable-overlap-schedule。

各阶段
  E0  环境检查：runbook 的 C0 那几项（bf16 checkpoint；代码包含基底 D，也就包含了 A 和 B），
      再加上：FP8 checkpoint 有 131 个 safetensors 文件，主机可用内存不少于 128 GiB，工作区
      没有未提交的改动。不满足就停。

  任何一部分失败都不阻断后面的部分。
  Part A（基底 A，bf16）
  A1  启动：加载摘要逐字等于 "consumed=1163, skipped=495, missing=0, unexpected=0"，日志里有
      "N-gram table: 128/128 shards"，哈希校验没有被跳过；再发一个请求，能正常出 token、
      logprob 里没有 NaN。
  A2  cookbook §3 的三个请求：关思考的对话；开思考的流式推理，思考和答案分开推送、答案里有
      36；工具调用，解析出 get_weather、地点是 Beijing。三段输出原样存进 a2_cookbook.md。
  A3  MMLU 冒烟：参数和仓库测试 _run_mmlu_smoke 一致（100 题、thinking、max_tokens 32768）。
      高于 0.70 判 PASS；低于 0.30 判 WARN（先查请求是否被服务端拒绝）；其余判 FAIL。
  A4  压测三组，只记录：vLLM recipe 的负载（随机 1024 入 / 256 出、256 个请求、并发 64），
      以及 512 入 / 128 出 × 100、8192 入 / 128 出 × 32（并发 8）。
  A5  在 vLLM 那组负载下抓几步 decode 的 profile（并发 64）。

  Part B（基底 B，bf16）
  B1  启动，判定同 A1。
  B2  MMLU 冒烟，判定同 A3，并和 A3 逐题对比。稠密分支和稀疏注意力在预算以内数学上等价，
      两边的差应在采样噪声以内。
  B3  同 A4 的三组压测。和 A4 的差就是稠密分支的收益。
  B4  同 A5。

  Part C（基底 B，FP8）
  C1  FP8 MoE 调参：bench_fused_moe --tune-block-config，FP8 权重、量化分块 128，覆盖 16–2048
      八个 token 档位，结果存成 moe_tuned_fp8.json。
  C2  调好的分块逐条做数值检查（FP8 权重、量化分块 128、512 专家、top-10、2560 / 640、ep=8、
      atol / rtol 5e-2）。没过的再用仓库 FP8 测试的容差 2e-1 复查一次：只在 2e-1 下通过的照样
      写进去，C2 判 WARN，报告里单独列出；2e-1 也没过的丢掉。通过的条目临时写进工作区的
      tuned_block_configs.py，git diff 存成 c2_tuned_configs.diff。没有条目通过时，C3 起用默认
      分块。
  C3  启动：加载摘要逐字等于 "consumed=148523, skipped=3566, missing=0, unexpected=0"，日志里有
      "N-gram table: 128/128 shards" 和 47.7 GiB 的表；C2 写进了条目时，日志里要有
      "Using tuned block config"；再发一个请求。
  C4  MMLU 冒烟，判定同 A3，并和 A3 逐题对比。
  C5  GPQA-Diamond：seed 17 / 23 / 41 / 59 / 97，模型卡推荐的 thinking 采样参数，max_tokens
      65536。只记录：每个 seed 的分数、中位数，和 bf16 的 0.904、模型卡 0.917 的差。每个
      seed 跑完就落盘。C3 没过或者 C4 低于 0.30 时不跑。
  C6  同 A4 的三组压测。vLLM 那组的输出吞吐和 vLLM 的 1430 tok/s 对比，这是追平的主指标。
  C7  decode profile 同 A5；停掉服务后，用 bench_one_batch --profile 抓 prefill：2 × 1024 token
      和 1 × 2048 token 各一次，每次都重新加载模型。

  Part D（基底 D，FP8，开 overlap）。切过去之后把 C2 通过的条目重新写进调优表。
  D1  启动：加载摘要同 C3；日志里有 "N-gram table: 128/128 shards" 和 "N-gram table placed in
      HBM: 320,001,536 rows x 160 split over 8 devices, 6.0 GiB each"，没有 "Overlap scheduler is
      disabled"；C2 写进了条目时要有 "Using tuned block config"；再发一个请求。KV 池的容量
      （token 数）和 C3 的并排记下来，差值就是表占掉的显存。
  D2  MMLU 冒烟，判定同 A3，并和 C4 逐题对比：这是开 overlap、在设备上查表这条路径的正确性检查，
      两边的差应在采样噪声以内。
  D3  同 C6 的三组压测。和 C6 的差就是这一步的收益。
  D4  同 A5。
  D5  bf16 checkpoint 启动，只做启动检查：加载摘要同 A1，日志里有 "... split over 8 devices,
      11.9 GiB each"，冒烟请求正常，记下 KV 池容量。
  Part D 的阶段失败时，报告里附上服务日志的最后几十行。

  评测都放在子进程里跑，同时盯着服务进程：服务一退出，评测就停下来，当前阶段记 FAIL。
  单个请求最多等 6 小时，超时就记下来、不重发；有请求出错时阶段判 WARN，出错数写进报告。

状态
  PASS 通过；WARN 能跑但有可疑之处；FAIL 失败；未运行：这次没选这个阶段，或者前面的结果已经
  说明它没有意义。
  E0 的 FAIL 是本机环境问题：按列出的问题修好后重跑，这种情况不打包、不用发回。
  其余阶段不管什么结果都不用自己排查，把两个压缩包原样发回。

时长：每次启动 20–30 分钟；Part A、B 各约 1 小时；C1 / C2 约 2–3 小时；GPQA 5 个 seed
估计 4–8 小时；D1–D4 约 1.5–2 小时，D5 约 0.5 小时。合计一个半到两个晚上，放在 tmux / screen
里或者用 nohup 跑。

用法：在 sglang-jax 仓库根目录下运行。
  python scripts/runbook_flashnext_v7x_parity.py --model-path /data/Qwen3.8-Flash-Next \\
      --fp8-model-path /data/Qwen3.8-Flash-Next-FP8
  python scripts/runbook_flashnext_v7x_parity.py ... --stages C5 --gpqa-seeds 59,97   # 断点重跑
  python scripts/runbook_flashnext_v7x_parity.py ... --dry-run    # 只打印要执行的命令
中途被打断后重跑之前，先确认没有残留的服务进程占着 TPU（pkill -f sgl_jax.launch_server）。
如果脚本没来得及切回原来的分支（比如被 kill -9），先在仓库根目录执行
  git checkout -- . && git checkout bringup/qwen4-exp-v7x

跑完把 <out>.tar.gz 和 <out>-profile.tar.gz 发回来（后者是 profile trace，体积较大）。
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import signal
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import runbook_flashnext_v7x as rb  # noqa: E402
import runbook_flashnext_v7x_followup as fu  # noqa: E402
import runbook_flashnext_v7x_stage2 as s2  # noqa: E402

PARTS = {
    "A": ("A1", "A2", "A3", "A4", "A5"),
    "B": ("B1", "B2", "B3", "B4"),
    "C": ("C1", "C2", "C3", "C4", "C5", "C6", "C7"),
    "D": ("D1", "D2", "D3", "D4", "D5"),
}
STAGES = ("E0", *PARTS["A"], *PARTS["B"], *PARTS["C"], *PARTS["D"])
STAGE_INFO = {
    "E0": "环境检查：TPU 设备、两份 checkpoint、仓库和代码版本、主机内存、工作区干净",
    "A1": "基底 A + bf16 启动（cookbook 命令）：加载摘要、n-gram 表、出 token",
    "A2": "cookbook §3 的三个调用示例",
    "A3": "MMLU 冒烟（基底 A，bf16）",
    "A4": "压测三组（基底 A，bf16）",
    "A5": "decode profile，vLLM 负载（基底 A，bf16）",
    "B1": "基底 B + bf16 启动：加载摘要、n-gram 表、出 token",
    "B2": "MMLU 冒烟（基底 B，bf16），和 A3 逐题对比",
    "B3": "压测三组（基底 B，bf16）",
    "B4": "decode profile，vLLM 负载（基底 B，bf16）",
    "C1": "FP8 MoE 调参：八个 token 档位的分块",
    "C2": "调好的 FP8 分块做数值检查，通过的临时写进工作区",
    "C3": "基底 B + FP8 启动：加载摘要、n-gram 表、调好的分块、出 token",
    "C4": "MMLU 冒烟（FP8），和 A3 逐题对比",
    "C5": "GPQA-Diamond × 5 个 seed（FP8，max_tokens 65536）",
    "C6": "压测三组（FP8）",
    "C7": "decode profile（vLLM 负载）+ prefill profile（bench_one_batch）（FP8）",
    "D1": "基底 D + FP8 启动，开 overlap：表在 HBM、overlap 没被关、KV 池容量",
    "D2": "MMLU 冒烟（基底 D，FP8），和 C4 逐题对比",
    "D3": "压测三组（基底 D，FP8）",
    "D4": "decode profile，vLLM 负载（基底 D，FP8）",
    "D5": "基底 D + bf16 启动，开 overlap：只看表放不放得下",
}
rb.STAGE_INFO.update(STAGE_INFO)  # stage_launch 和 stage2 的函数打印阶段标题时查这张表

# 每部分的代码基底。D 是 rb.REQUIRED_COMMIT，A、B 是它的祖先
BASES = {
    "A": "429ed463",  # sgl-project/sglang-jax#1781 的 head
    "B": "8fe3d481",  # A 之上：预算以内走稠密注意力，加载 FP8 checkpoint
    "D": "b7f05c28",  # B 之上：N-gram 表放进 HBM、在设备上查表，overlap 调度不再关掉
}
PART_BASE = {"A": "A", "B": "B", "C": "B", "D": "D"}
PART_LABEL = {
    "A": "A：基底 A + bf16",
    "B": "B：基底 B + bf16",
    "C": "C：基底 B + FP8",
    "D": "D：基底 D + FP8，开 overlap",
}

MIN_MEM_AVAILABLE_GIB = 128  # bf16 的 n-gram 表 95.4 GiB，FP8 的 47.7 GiB，两次启动不同时在内存里

# 三次启动共用，和 cookbook §2.3 一致
LAUNCH = dict(s2.LAUNCH_PLE_ON)
COOKBOOK_PARSERS = ("--reasoning-parser", "qwen3", "--tool-call-parser", "qwen3_coder")
LAUNCH_COOKBOOK = {**LAUNCH, "extra_args": COOKBOOK_PARSERS}
EXPECTED_LOAD_SUMMARY_FP8 = "consumed=148523, skipped=3566, missing=0, unexpected=0"
NGRAM_ALLOCATED_FP8 = "N-gram table allocated: 320,001,536 rows x 160 = 47.7 GiB"
LAUNCH_OVERLAP = {**LAUNCH, "overlap": True}
NGRAM_IN_HBM = "N-gram table placed in HBM: 320,001,536 rows x 160 split over 8 devices, "
NGRAM_IN_HBM_FP8 = NGRAM_IN_HBM + "6.0 GiB each"
NGRAM_IN_HBM_BF16 = NGRAM_IN_HBM + "11.9 GiB each"
OVERLAP_DISABLED = "Overlap scheduler is disabled"
LOG_TAIL_LINES = 40

# vLLM recipe 的负载：4 × H100、FP8 时输出吞吐约 1430 tok/s
VLLM_BENCH = {"input_len": 1024, "output_len": 256, "num_prompts": 256, "concurrency": 64}
VLLM_OUTPUT_THROUGHPUT = 1430
BENCH_SWEEP = [VLLM_BENCH, *fu.BENCH_SWEEP]

GPQA_BF16_MEDIAN = 0.904  # bf16、PLE 打开时 5 个 seed 的中位数
GPQA_TOLERANCE = 0.015  # 精度目标：FP8 的 GPQA 中位数和 bf16 相差不超过这么多

MOE_TOKENS_FP8 = (16, 32, 64, 128, 256, 512, 1024, 2048)
MOE_QUANT_FP8 = {"w_dtype": "float8_e4m3fn", "quant_block_k": 128}
MOE_TUNE_FP8_ARGS = [
    "-m", "benchmark.moe.bench_fused_moe", "--tune-block-config",
    # FP8 默认的量化分块是 256，intermediate 640 不能被它整除
    "--weight-dtype", "float8_e4m3fn", "--quant-block-k", "128",
    "--num-experts", "512", "--top-k", "10", "--hidden-size", "2560", "--intermediate-size", "640",
    "--num-tokens", *map(str, MOE_TOKENS_FP8),
    "--bf-candidates", "128", "640",
    "--bd-candidates", "256", "512", "640", "1280", "2560",
    "--imbalance-mode", "dirichlet",
    "--max-configs", "20",
]  # fmt: skip
MOE_TUNED_FP8 = "moe_tuned_fp8.json"
# 5e-2 下没过时的复查容差，和仓库 FP8 sub-channel 测试相同。参照实现用的是同一份量化权重和
# scale，差别只来自累加顺序；分块配置真坏了误差是 O(1)，这个容差照样能发现
MOE_RECHECK_TOL = 2e-1

# (batch, 每个请求的 token 数)。服务端一个 prefill 批次最多 2048 个 token：vLLM 那组负载是两个
# 1024 token 的请求，长 prompt 是一个 2048 的 chunk
PREFILL_PROFILES = ((2, 1024), (1, 2048))
PREFILL_TIMEOUT = 3600  # 每次都要重新加载模型
PREFILL_DIR = "profile_c7_prefill"

COOKBOOK_TIMEOUT = 1800
COOKBOOK_WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
            },
            "required": ["location"],
        },
    },
}


# ---- 代码基底 -----------------------------------------------------------------


def resolve(args, rev: str) -> str | None:
    r = rb.git(args.repo, "rev-parse", "--verify", "--quiet", f"{rev}^{{commit}}")
    return r.stdout.strip() if r.returncode == 0 else None


def current_ref(args) -> str:
    branch = rb.git(args.repo, "symbolic-ref", "--quiet", "--short", "HEAD").stdout.strip()
    return branch or rb.git(args.repo, "rev-parse", "HEAD").stdout.strip()


def checkout(run: rb.Run, args, part: str) -> None:
    """把仓库切到这一部分的基底，核对 HEAD。调优表的临时改动先还原。"""
    name = PART_BASE[part]
    commit = BASES[name]
    rb.git(args.repo, "checkout", "HEAD", "--", s2.TUNED_CONFIGS_FILE)
    r = rb.git(args.repo, "checkout", "--detach", commit)
    want, head = resolve(args, commit), rb.git(args.repo, "rev-parse", "HEAD").stdout.strip()
    if r.returncode != 0 or want is None or head != want:
        why = r.stderr.strip() or f"HEAD 是 {head}"
        raise RuntimeError(f"切到基底 {name}（{commit}）失败：{why}")
    log = run.out / "checkouts.json"
    entries = json.loads(log.read_text()) if log.exists() else []
    entries.append({"part": part, "base": name, "commit": commit, "head": head})
    log.write_text(json.dumps(entries, indent=2))
    run.log(f"Part {part}：代码切到基底 {name}，HEAD {head[:12]}")


def restore(run: rb.Run, args, original: str) -> None:
    rb.git(args.repo, "checkout", "HEAD", "--", s2.TUNED_CONFIGS_FILE)
    r = rb.git(args.repo, "checkout", original)
    if r.returncode == 0:
        run.log(f"代码已切回 {original}")
    else:
        run.log(
            f"切回 {original} 失败：{r.stderr.strip()}。请在仓库根目录执行 "
            f"git checkout -- . && git checkout {original}"
        )


def copy_scripts(run: rb.Run) -> None:
    """切换基底会把 scripts/ 下的这几份脚本换掉；评测和数值检查的子进程从输出目录里的副本启动。"""
    dst = run.out / "scripts"
    dst.mkdir(exist_ok=True)
    for path in (rb.__file__, fu.__file__, s2.__file__, __file__):
        shutil.copy2(path, dst)
    s2.WORKER_SCRIPT = (dst / Path(s2.__file__).name).resolve()


# ---- E0 -----------------------------------------------------------------------


def stage_e0(run: rb.Run, args) -> bool:
    problems = []
    fp8 = Path(args.fp8_model_path)
    n_st = len(list(fp8.glob("*.safetensors"))) if fp8.is_dir() else 0
    if not fp8.is_dir():
        problems.append(f"--fp8-model-path 不是已有的目录：{fp8}")
    elif not (fp8 / "config.json").exists():
        problems.append("--fp8-model-path 目录下没有 config.json")
    if n_st != rb.EXPECTED_SAFETENSORS:
        problems.append(
            f"FP8 checkpoint 期望 {rb.EXPECTED_SAFETENSORS} 个 safetensors 文件，实际 {n_st} 个"
        )
    if Path(args.repo, "python", "sgl_jax").is_dir():
        if rb.git(args.repo, "status", "--porcelain", "--untracked-files=no").stdout.strip():
            problems.append(
                "工作区有未提交的改动，脚本要切换代码，工作区必须干净。这些改动通常是之前的测试"
                "临时写进去的（git diff 可以看），确认不需要后在仓库根目录执行 git checkout -- ."
            )
    return rb.stage_c0(run, args, "E0", MIN_MEM_AVAILABLE_GIB, problems)


# ---- 各阶段 -------------------------------------------------------------------


def cookbook_requests(run: rb.Run, srv: rb.Server, args) -> None:
    """cookbook §3 的三个请求，代码和文档里的示例一致，只把输出记下来、检查一遍。"""
    from openai import OpenAI

    client = OpenAI(
        base_url=f"{srv.base}/v1", api_key="EMPTY", timeout=COOKBOOK_TIMEOUT, max_retries=0
    )
    model = args.model_path
    out, problems = {}, []

    def attempt(key, fn):
        try:
            fn()
        except Exception as e:  # noqa: BLE001 - 一个请求出错，另外两个照样发
            out[key] = {"error": f"{type(e).__name__}: {e}"}
            problems.append(f"{key} 请求出错：{type(e).__name__}: {e}")

    def chat():
        r = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "What is the capital of France?"}],
            max_tokens=256,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        msg = r.choices[0].message
        out["chat"] = {
            "content": msg.content,
            "reasoning_content": getattr(msg, "reasoning_content", None),
            "finish_reason": r.choices[0].finish_reason,
        }
        if not (msg.content or "").strip():
            problems.append("关思考的对话没有输出")
        if out["chat"]["reasoning_content"]:
            problems.append("关了思考，返回里仍有 reasoning_content")

    def reasoning():
        stream = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Solve step by step: what is 15% of 240?"}],
            temperature=1.0,
            top_p=0.95,
            max_tokens=4096,
            stream=True,
            extra_body={
                "top_k": 20,
                "min_p": 0.0,
                "repetition_penalty": 1.0,
                "chat_template_kwargs": {"enable_thinking": True},
            },
        )
        thinking, answer, order = [], [], []
        for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            if getattr(delta, "reasoning_content", None):
                thinking.append(delta.reasoning_content)
                order.append("r")
            if delta.content:
                answer.append(delta.content)
                order.append("c")
        content = "".join(answer)
        out["reasoning"] = {
            "reasoning_content": "".join(thinking),
            "content": content,
            "reasoning_chunks": len(thinking),
            "content_chunks": len(answer),
        }
        if not thinking:
            problems.append("流式推理没有推送 reasoning_content")
        if "c" in order and "r" in order[order.index("c") :]:
            problems.append("答案开始推送之后又出现了 reasoning_content")
        if "36" not in content:
            problems.append(f"推理的最终答案里没有 36：{content[-200:]!r}")
        if "<think>" in content or "</think>" in content:
            problems.append("答案里混进了 <think> 标签，思考没有被单独解析出来")

    def tool():
        r = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "What's the weather in Beijing?"}],
            tools=[COOKBOOK_WEATHER_TOOL],
            tool_choice="auto",
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        msg = r.choices[0].message
        calls = [(c.function.name, c.function.arguments) for c in msg.tool_calls or []]
        # 和 cookbook 示例的 print 输出相同
        printed = [f"{name} {arguments}" for name, arguments in calls] or [msg.content]
        out["tool"] = {"tool_calls": calls, "content": msg.content, "printed": printed}
        try:
            location = json.loads(calls[0][1]).get("location", "") if calls else ""
        except (json.JSONDecodeError, AttributeError):
            location = ""
        if not calls or calls[0][0] != "get_weather" or "Beijing" not in location:
            problems.append(f"工具调用没有解析出 get_weather / Beijing：{printed}")

    attempt("chat", chat)
    attempt("reasoning", reasoning)
    attempt("tool", tool)
    (run.out / "a2_cookbook.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    (run.out / "a2_cookbook.md").write_text(cookbook_markdown(out))
    for p in problems:
        run.log(f"  问题：{p}")
    run.record("A2", "FAIL" if problems else "PASS", problems=problems, outputs=out)


def cookbook_markdown(out: dict) -> str:
    """cookbook 里三处 TODO 要填的内容。"""
    chat, rsn, tool = out.get("chat", {}), out.get("reasoning", {}), out.get("tool", {})
    lines = ["# cookbook §3 的实测输出", "", "## 3.1 Basic Chat Completion", "", "```text"]
    lines += [str(chat.get("content", chat.get("error"))), "```", "", "## 3.2 Reasoning", ""]
    if "error" in rsn:
        lines += [f"出错：{rsn['error']}"]
    else:
        lines += [
            f"reasoning_content 推送了 {rsn.get('reasoning_chunks')} 个 chunk，"
            f"content 推送了 {rsn.get('content_chunks')} 个 chunk。最终答案：",
            "",
            "```text",
            str(rsn.get("content")),
            "```",
        ]
    lines += ["", "## 3.3 Tool Calling", "", "```text"]
    lines += tool.get("printed") or [str(tool.get("error"))]
    lines += ["```", ""]
    return "\n".join(lines)


def load_calls(run: rb.Run, name: str) -> dict:
    path = run.out / f"{name}.calls.jsonl"
    if not path.exists():
        return {}
    calls = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return {c["question"]: c for c in calls if c.get("question")}


def compare_mmlu(run: rb.Run, ref: str, name: str) -> dict | None:
    """两次 MMLU 逐题对比：都对、只有参照对、只有这次对、都不对，以及抽到的答案相同的题数。"""
    a, b = load_calls(run, ref), load_calls(run, name)
    common = sorted(set(a) & set(b))
    if not common:
        return None

    def right(c):
        return c.get("pred") is not None and c.get("pred") == c.get("answer")

    pairs = [(right(a[q]), right(b[q])) for q in common]
    return {
        "against": ref,
        "questions": len(common),
        "both_right": sum(x and y for x, y in pairs),
        "only_ref_right": sum(x and not y for x, y in pairs),
        "only_this_right": sum(y and not x for x, y in pairs),
        "both_wrong": sum(not x and not y for x, y in pairs),
        "same_answer": sum(a[q].get("pred") == b[q].get("pred") for q in common),
    }


def eval_name(stage: str) -> str:
    return f"{stage.lower()}_mmlu"


def mmlu_step(run: rb.Run, args, stage: str, ref: str | None = None):
    def fn(srv, ready, checks):
        job = s2.MMLU_JOB
        res = s2.run_eval_job(run, srv, args, eval_name(stage), job, job["seed"])
        details = {"mmlu": res}
        if ref:
            details["vs"] = compare_mmlu(run, eval_name(ref), eval_name(stage))
            run.log(f"  和 {ref} 逐题对比：{details['vs']}")
        run.record(stage, s2.mmlu_status(run, res), **details)

    return fn


def bench_step(run: rb.Run, args, stage: str):
    def fn(srv, ready, checks):
        perf, errors = s2.measure_perf(run, srv, args, None, sweep=BENCH_SWEEP)
        run.record(stage, "FAIL" if not perf["bench"] else "WARN" if errors else "PASS", **perf)

    return fn


def capture_decode(run: rb.Run, srv: rb.Server, args) -> dict:
    return fu.capture_profile(run, srv, args, request=s2.PROFILE_DECODE, load=VLLM_BENCH)


def profile_step(run: rb.Run, args, stage: str):
    def fn(srv, ready, checks):
        profile = capture_decode(run, srv, args)
        run.log(f"  profile（{srv.label}）：{profile}")
        run.record(stage, "PASS", profile=profile)

    return fn


def launch_step(
    run: rb.Run, stage: str, summary: str, written_from=None, required=(), forbidden=()
):
    def fn(srv, ready, checks):
        written = run.results.get(written_from, {}).get("written", []) if written_from else []
        s2.check_launch(run, stage, srv, ready, checks, summary, written, required, forbidden)

    return fn


def prefill_cmd(args, batch: int, input_len: int, out: Path) -> list[str]:
    """bench_one_batch 收和 launch_server 相同的参数，再加上 batch 的形状。"""
    cmd = rb.server_cmd(args, "qsa_sparse", LAUNCH)
    cmd[cmd.index("sgl_jax.launch_server")] = "sgl_jax.bench_one_batch"
    return cmd + [
        "--batch-size", str(batch),
        "--input-len", str(input_len),
        "--output-len", "1",
        "--profile",
        "--profile-filename-prefix", str((out / "prefill").resolve()),
        "--result-filename", str((out.parent / "c7_bench_one_batch.jsonl").resolve()),
    ]  # fmt: skip


def capture_prefill(run: rb.Run, args) -> tuple[list[dict], dict | None]:
    out = run.out / PREFILL_DIR
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir()
    runs = []
    for batch, input_len in PREFILL_PROFILES:
        cmd = prefill_cmd(args, batch, input_len, out)
        run.log(f"  bench_one_batch：{shlex.join(cmd)}")
        log = run.out / f"c7_prefill_b{batch}_in{input_len}.log"
        rc = rb.run_logged(args, cmd, log, timeout=PREFILL_TIMEOUT)
        time.sleep(30)  # 等 TPU 运行时释放芯片，再进行下一次加载
        trace = out / f"prefill_batch{batch}_input{input_len}_output1.tb"
        captured = rc == 0 and any(trace.rglob("*.xplane.pb"))
        run.log(
            f"  prefill {batch} × {input_len}：返回码 {rc}，{'抓到' if captured else '没抓到'} trace"
        )
        runs.append(
            {"batch": batch, "input_len": input_len, "returncode": rc, "captured": captured}
        )
    summary = fu.prune_profile(out) if any(r["captured"] for r in runs) else None
    return runs, summary


# ---- 三部分 -------------------------------------------------------------------


def part_a(run: rb.Run, args, wanted: set[str]) -> None:
    steps = [
        ("A1", launch_step(run, "A1", s2.EXPECTED_LOAD_SUMMARY_PLE)),
        ("A2", lambda srv, ready, checks: cookbook_requests(run, srv, args)),
        ("A3", mmlu_step(run, args, "A3")),
        ("A4", bench_step(run, args, "A4")),
        ("A5", profile_step(run, args, "A5")),
    ]
    steps = [(s, fn) for s, fn in steps if s in wanted]
    rb.stage_launch(run, args, "qsa_sparse", steps, LAUNCH_COOKBOOK, "a_bf16")


def part_b(run: rb.Run, args, wanted: set[str]) -> None:
    steps = [
        ("B1", launch_step(run, "B1", s2.EXPECTED_LOAD_SUMMARY_PLE)),
        ("B2", mmlu_step(run, args, "B2", ref="A3")),
        ("B3", bench_step(run, args, "B3")),
        ("B4", profile_step(run, args, "B4")),
    ]
    steps = [(s, fn) for s, fn in steps if s in wanted]
    rb.stage_launch(run, args, "qsa_sparse", steps, LAUNCH, "b_bf16")


def fp8_args(args):
    return argparse.Namespace(**{**vars(args), "model_path": args.fp8_model_path})


def rewrite_tuned(run: rb.Run, args) -> None:
    """把 C2 通过检查的条目再写进调优表：切换代码会把调优表还原。"""
    written = run.results.get("C2", {}).get("written", [])
    if not written:
        run.log("C2 没有通过检查的条目，服务用默认分块")
        return
    try:
        s2.write_tuned_entries(args.repo, written)
        run.log(f"按 C2 的结果，把 {len(written)} 条分块重新写进工作区的调优表")
    except Exception as e:  # noqa: BLE001 - 写不进去就用默认分块
        rb.git(args.repo, "checkout", "HEAD", "--", s2.TUNED_CONFIGS_FILE)
        run.log(f"C2 的分块写不进调优表，服务用默认分块：{type(e).__name__}: {e}")


def attach_log_tail(run: rb.Run, stages, label: str) -> None:
    """失败的阶段附上服务日志的最后 LOG_TAIL_LINES 行。"""
    path = run.out / f"server_{label}.log"
    if not path.exists():
        return
    tail = path.read_text(errors="replace").splitlines()[-LOG_TAIL_LINES:]
    for s in stages:
        if run.results.get(s, {}).get("status") == "FAIL":
            run.results[s]["log_tail"] = tail
    run.save()


def part_c(run: rb.Run, args, wanted: set[str]) -> None:
    args = fp8_args(args)
    tuning = {
        "C1": lambda: s2.tune_moe(
            run, args, "C1", MOE_TUNE_FP8_ARGS, MOE_TOKENS_FP8, MOE_TUNED_FP8
        ),
        "C2": lambda: s2.check_tuned(
            run, args, "C2", MOE_TUNED_FP8, MOE_QUANT_FP8, MOE_RECHECK_TOL
        ),
    }
    for stage, fn in tuning.items():
        if stage in wanted:
            try:
                fn()
            except Exception as e:  # noqa: BLE001 - 调参失败不阻断后面，服务用默认分块
                run.log(f"{stage} 出错：{type(e).__name__}: {e}")
                run.record(stage, "FAIL", error=f"{type(e).__name__}: {e}")
    if "C2" not in wanted:  # 断点重跑：用上次 C2 的结果
        rewrite_tuned(run, args)

    decode = {}

    def c7(srv, ready, checks):
        try:
            decode["profile"] = capture_decode(run, srv, args)
            run.log(f"  profile（{srv.label}）：{decode['profile']}")
        except Exception as e:  # noqa: BLE001 - 停掉服务后照样抓 prefill
            decode["error"] = f"{type(e).__name__}: {e}"
        run.results["C7"] = {"status": "未完成", "decode": decode}
        run.save()

    steps = [
        ("C3", launch_step(run, "C3", EXPECTED_LOAD_SUMMARY_FP8, "C2", (NGRAM_ALLOCATED_FP8,))),
        ("C4", mmlu_step(run, args, "C4", ref="A3")),
        ("C5", lambda srv, ready, checks: s2.run_gpqa(
            run, srv, args, "C5", ("C3", "C4"), fresh="C3" in wanted
        )),
        ("C6", bench_step(run, args, "C6")),
        ("C7", c7),
    ]  # fmt: skip
    steps = [(s, fn) for s, fn in steps if s in wanted]
    if steps:
        rb.stage_launch(run, args, "qsa_sparse", steps, LAUNCH, "c_fp8")
    if "C7" not in wanted:
        return
    if not decode:  # 服务没起来或者在 C7 之前退出了
        decode["error"] = run.results.get("C7", {}).get("error", "没有运行")
    run.log("===== C7（续）：停掉服务后用 bench_one_batch 抓 prefill")
    prefill, summary = capture_prefill(run, args)
    got = ["profile" in decode] + [r["captured"] for r in prefill]
    status = "PASS" if all(got) else "WARN" if any(got) else "FAIL"
    run.record("C7", status, decode=decode, prefill=prefill, prefill_profile=summary)


def part_d(run: rb.Run, args, wanted: set[str]) -> None:
    fp8 = fp8_args(args)
    rewrite_tuned(run, args)  # 和 C 用同一套分块，C → D 的差别只剩这一步
    steps = [
        ("D1", launch_step(
            run, "D1", EXPECTED_LOAD_SUMMARY_FP8, "C2", (NGRAM_IN_HBM_FP8,), (OVERLAP_DISABLED,)
        )),
        ("D2", mmlu_step(run, fp8, "D2", ref="C4")),
        ("D3", bench_step(run, fp8, "D3")),
        ("D4", profile_step(run, fp8, "D4")),
    ]  # fmt: skip
    steps = [(s, fn) for s, fn in steps if s in wanted]
    if steps:
        rb.stage_launch(run, fp8, "qsa_sparse", steps, LAUNCH_OVERLAP, "d_fp8")
        attach_log_tail(run, [s for s, _ in steps], "d_fp8")
    if "D5" in wanted:
        d5 = launch_step(run, "D5", s2.EXPECTED_LOAD_SUMMARY_PLE, required=(NGRAM_IN_HBM_BF16,))
        rb.stage_launch(run, args, "qsa_sparse", [("D5", d5)], LAUNCH_OVERLAP, "d_bf16")
        attach_log_tail(run, ["D5"], "d_bf16")


PART_FNS = {"A": part_a, "B": part_b, "C": part_c, "D": part_d}


def run_part(run: rb.Run, args, part: str, wanted: set[str]) -> None:
    stages = [s for s in PARTS[part] if s in wanted]
    if not stages:
        return
    try:
        checkout(run, args, part)
        PART_FNS[part](run, args, set(stages))
    except Exception as e:  # noqa: BLE001 - 这一部分出错，后面的部分照样跑
        run.log(f"Part {part} 出错：{type(e).__name__}: {e}")
        for s in stages:
            if run.results.get(s, {}).get("status") in (None, "未完成"):
                run.record(s, "FAIL", error=f"没有运行完：{type(e).__name__}: {e}")


# ---- 报告 ---------------------------------------------------------------------


def workload(cfg: dict) -> str:
    return (
        f"{cfg['input_len']} 入 / {cfg['output_len']} 出 × {cfg['num_prompts']}，"
        f"并发 {cfg['concurrency']}"
    )


def bench_of(r: dict, stage: str, cfg: dict) -> dict | None:
    return r.get(stage, {}).get("bench", {}).get(str(cfg["input_len"]))


def write_report(run: rb.Run, args, repo: dict) -> Path:
    r = run.results
    perf_stage = {"A": "A4", "B": "B3", "C": "C6", "D": "D3"}
    mmlu_stage = {"A": "A3", "B": "B2", "C": "C4", "D": "D2"}
    checkouts_path = run.out / "checkouts.json"
    checkouts = json.loads(checkouts_path.read_text()) if checkouts_path.exists() else []
    lines = ["# Qwen3.8-Flash-Next 性能追平测试（R1）", "", "## 1. 设计", ""]
    lines += ["| 组合 | 代码 | checkpoint |", "| --- | --- | --- |"]
    for part, label in PART_LABEL.items():
        heads = sorted({c["head"][:12] for c in checkouts if c["part"] == part})
        ckpt = {"C": "官方 FP8", "D": "官方 FP8（D5 用 bf16，只做启动检查）"}.get(part, "bf16")
        lines += [
            f"| {label} | {BASES[PART_BASE[part]]}，HEAD {', '.join(heads) or '-'} | {ckpt} |"
        ]
    lines += [
        "",
        "启动参数（各部分相同，C、D 换成 FP8 checkpoint，D 去掉 --disable-overlap-schedule，"
        f"A1 另加 {' '.join(COOKBOOK_PARSERS)}）：",
        "",
        "```bash",
        shlex.join(rb.server_cmd(args, "qsa_sparse", LAUNCH)),
        "```",
        "",
        "| 阶段 | 结果 | 测什么 |",
        "| --- | --- | --- |",
    ]
    lines += [
        f"| {s} | {r.get(s, {}).get('status', rb.NOT_RUN)} | {STAGE_INFO[s]} |" for s in STAGES
    ]

    lines += ["", "## 2. 结果与官方对比", "", "### 2.1 吞吐", ""]
    lines += [
        "| 负载 | 组合 | 完成数 | TTFT 中位数 ms | TPOT 中位数 ms | 输出吞吐 tok/s | 总吞吐 tok/s |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    keys = (
        "completed",
        "median_ttft_ms",
        "median_tpot_ms",
        "output_throughput",
        "total_throughput",
    )
    for cfg in BENCH_SWEEP:
        for part, stage in perf_stage.items():
            m = bench_of(r, stage, cfg)
            cells = " | ".join(s2.fmt(m.get(k, "-")) for k in keys) if m else "- | - | - | - | -"
            lines += [f"| {workload(cfg)} | {PART_LABEL[part]} | {cells} |"]
        if cfg is VLLM_BENCH:
            lines += [
                f"| {workload(cfg)} | vLLM 4 × H100，FP8（recipe） | | | | {VLLM_OUTPUT_THROUGHPUT} | |"
            ]

    lines += ["", "### 2.2 精度", ""]
    lines += [
        "| 评测 | 分数 | 请求数 | 截断 | 没抽到答案 | 被拒 | 超时 | 出错 | 输出 token p50 / p90 / max |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    evals = [(f"MMLU（{PART_LABEL[p]}）", r.get(s, {}).get("mmlu")) for p, s in mmlu_stage.items()]
    evals += [
        (f"GPQA seed {seed}（FP8）", res) for seed, res in r.get("C5", {}).get("runs", {}).items()
    ]
    for name, e in evals:
        if not e:
            continue
        if "score" not in e:
            lines += [f"| {name} | 出错：{e.get('error')} | | | | | | | |"]
            continue
        t = e.get("output_tokens") or {}
        counts = [
            e.get(k, "-")
            for k in ("requests", "finish_length", "pred_none", "rejected", "timeouts", "errors")
        ]
        tokens = " / ".join(str(t.get(k, "-")) for k in ("p50", "p90", "max"))
        lines += [
            f"| {name} | {e['score']:.4f} | " + " | ".join(map(str, counts)) + f" | {tokens} |"
        ]
    c5 = r.get("C5", {})
    if "median" in c5:
        lines += [
            "",
            f"GPQA-Diamond（FP8）中位数 {c5['median']}：和 bf16 的 {GPQA_BF16_MEDIAN} 差 "
            f"{c5['median'] - GPQA_BF16_MEDIAN:+.4f}，和模型卡 {s2.GPQA_PUBLISHED} 差 {c5['delta']:+.4f}。"
            "seed 只用来区分 5 次运行：服务端不使用请求里的 seed，重跑同一个 seed 不会复现。",
        ]
    vs = [(s, r.get(s, {}).get("vs")) for s in ("B2", "C4", "D2")]
    vs = [(s, v) for s, v in vs if v]
    if vs:
        lines += [
            "",
            "MMLU 逐题对比：",
            "",
            "| 阶段 | 参照 | 题数 | 都对 | 只有参照对 | 只有这次对 | 都错 | 抽到的答案相同 |",
            "| --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
        lines += [
            f"| {s} | {v['against'].split('_')[0].upper()} | {v['questions']} | {v['both_right']} | "
            f"{v['only_ref_right']} | {v['only_this_right']} | {v['both_wrong']} | {v['same_answer']} |"
            for s, v in vs
        ]

    launches = [(s, r[s]) for s in ("A1", "B1", "C3", "D1", "D5") if "load_summary" in r.get(s, {})]
    if launches:
        lines += [
            "",
            "### 2.3 启动",
            "",
            "| 阶段 | 结果 | 加载摘要 | n-gram 表 | KV 池 token 数 |",
            "| --- | --- | --- | --- | --- |",
        ]
        for s, c in launches:
            table = c.get("ngram_in_hbm") or c.get("ngram_allocated") or "-"
            summary = "符合" if c.get("load_summary_ok") else f"`{c.get('load_summary')}`"
            lines += [
                f"| {s} | {c['status']} | {summary} | {table.split('] ')[-1]} | "
                f"{c.get('kv_tokens', '-')} |"
            ]
        c3, d1 = r.get("C3", {}).get("kv_tokens"), r.get("D1", {}).get("kv_tokens")
        if c3 and d1:
            lines += [
                "",
                f"KV 池 D1 比 C3 少 {c3 - d1} 个 token，这就是 n-gram 表在 HBM 里占掉的部分。",
            ]

    lines += ["", "## 3. 结论（脚本按目标自动判定）", ""]
    lines += conclusions(r)
    lines += ["", "## 4. 优化贡献占比（vLLM 负载的输出吞吐）", ""]
    lines += contributions(r)
    lines += [
        "",
        "## 5. 后续优化 / 6. 后续测试目的",
        "",
        "回传后结合第 2–4 节和 profile 包写进正式报告。",
    ]

    lines += ["", "## 附录", ""]
    for s in STAGES:
        c = r.get(s, {})
        lines += [f"- {s} 问题：{p}" for p in c.get("problems", [])]
        lines += [f"- {s}：{c[k]}" for k in ("error", "reason") if k in c]
        lines += [f"- {s} 性能测量出错：{e}" for e in c.get("perf_errors", [])]
    for s in PARTS["D"]:
        tail = r.get(s, {}).get("log_tail")
        if tail:
            lines += ["", f"{s} 失败时服务日志的最后 {len(tail)} 行：", "", "```text", *tail, "```"]
    if (run.out / "a2_cookbook.md").exists():
        lines += ["", "cookbook 三个请求的输出见 a2_cookbook.md。"]
    c2 = r.get("C2", {}).get("checks", [])
    if c2:
        lines += [
            "",
            "### C1 / C2 调好的 FP8 分块",
            "",
            f"5e-2 没过的条目用 {MOE_RECHECK_TOL:g} 复查；误差是 5e-2 下 assert_allclose 报的超差元素里的最大值。",
            "",
            "| token 数 | 分块 | 数值检查 | 最大绝对误差 | 最大相对误差 |",
            "| --- | --- | --- | --- | --- |",
        ]
        verdict = {
            s2.MOE_CHECK_TOL: "5e-2 下通过",
            MOE_RECHECK_TOL: "只在 2e-1 下通过",
            None: "没通过",
        }
        lines += [
            f"| {c['num_tokens']} | {tuple(c['value'])} | {verdict[c.get('tol')]} | "
            f"{c.get('max_abs_diff', '-')} | {c.get('max_rel_diff', '-')} |"
            for c in c2
        ]
        loose = [c["num_tokens"] for c in c2 if c.get("tol") == MOE_RECHECK_TOL]
        if loose:
            lines += ["", f"只在 2e-1 下通过、照样写进调优表的条目（token 数）：{loose}"]
        diff = run.out / "c2_tuned_configs.diff"
        if diff.exists():
            lines += ["", "写进工作区的改动：", "", "```diff", diff.read_text().rstrip(), "```"]
    if r.get("C1", {}).get("missing_tokens"):
        lines += ["", f"C1 没调出条目的 token 数：{r['C1']['missing_tokens']}"]
    for s in ("A5", "B4", "D4"):
        if r.get(s, {}).get("profile"):
            lines += [f"- {s} profile：{r[s]['profile']}"]
    c7 = r.get("C7", {})
    if c7.get("decode") or c7.get("prefill"):
        lines += [f"- C7 decode profile：{c7.get('decode')}", f"- C7 prefill：{c7.get('prefill')}"]
    lines += [
        "",
        "工作区相对最后一个基底的全部改动在 workspace.diff。CPU 单测在本地跑，不在这次运行里。",
    ]
    lines += ["", f"运行结束时仓库：{repo.get('branch')} @ {repo.get('head')}"]
    path = run.out / "report.md"
    path.write_text("\n".join(lines) + "\n")
    return path


def conclusions(r: dict) -> list[str]:
    lines = []
    for stage, part in (("C6", "C"), ("D3", "D")):
        m = bench_of(r, stage, VLLM_BENCH) or {}
        if "output_throughput" not in m:
            lines += [f"- 吞吐（{PART_LABEL[part]}）：{stage} 没有 vLLM 负载的数字。"]
            continue
        x = m["output_throughput"]
        verdict = "达到" if x >= VLLM_OUTPUT_THROUGHPUT else "没达到"
        lines += [
            f"- 吞吐（{PART_LABEL[part]}）：vLLM 负载下输出 {x:.1f} tok/s，是 vLLM "
            f"{VLLM_OUTPUT_THROUGHPUT} tok/s 的 {x / VLLM_OUTPUT_THROUGHPUT:.0%}，{verdict}追平目标。"
        ]
    for stage in ("C4", "D2"):
        m = r.get(stage, {}).get("mmlu", {})
        if "score" in m:
            verdict = "达到" if m["score"] > s2.MMLU_PASS else "没达到"
            lines += [f"- {stage} MMLU {m['score']:.2f}，{verdict}门槛 {s2.MMLU_PASS}。"]
    c5 = r.get("C5", {})
    if "median" in c5:
        d = c5["median"] - GPQA_BF16_MEDIAN
        verdict = "在" if abs(d) <= GPQA_TOLERANCE else "超出"
        lines += [
            f"- FP8 GPQA 中位数 {c5['median']}，和 bf16 的 {GPQA_BF16_MEDIAN} 差 {d:+.4f}，"
            f"{verdict} ±{GPQA_TOLERANCE} 以内。"
        ]
    noise = "（100 题、准确率 0.9 左右时，两次运行之差的标准差约 0.04）"
    for this, ref, what in (("B2", "A3", "稠密分支"), ("D2", "C4", "表在 HBM + overlap")):
        a, b = r.get(ref, {}).get("mmlu", {}), r.get(this, {}).get("mmlu", {})
        if "score" in a and "score" in b:
            lines += [f"- {what}：{this} − {ref} = {b['score'] - a['score']:+.2f}{noise}。"]
    return lines


def contributions(r: dict) -> list[str]:
    tput = {
        p: (bench_of(r, s, VLLM_BENCH) or {}).get("output_throughput")
        for p, s in (("A", "A4"), ("B", "B3"), ("C", "C6"), ("D", "D3"))
    }
    if any(tput[p] is None for p in "ABC"):
        return ["A4 / B3 / C6 有一组缺 vLLM 负载的数字，算不出占比。"]
    steps = [
        ("预算以内走稠密注意力（A → B）", "A", "B"),
        ("FP8 checkpoint + FP8 MoE 分块（B → C）", "B", "C"),
    ]
    if tput["D"] is not None:
        steps += [("N-gram 表进 HBM + overlap（C → D）", "C", "D")]
    last = steps[-1][2]
    a, end = tput["A"], tput[last]
    lines = [
        " → ".join(f"{p} {tput[p]:.1f}" for p in "ABCD" if tput[p] is not None)
        + f" tok/s；vLLM {VLLM_OUTPUT_THROUGHPUT}。"
    ]
    if end > a:
        lines += ["", f"| 优化 | 提升 tok/s | 占 A → {last} 的比例 |", "| --- | --- | --- |"]
        lines += [
            f"| {name} | {tput[y] - tput[x]:+.1f} | {(tput[y] - tput[x]) / (end - a):.0%} |"
            for name, x, y in steps
        ]
    gap = VLLM_OUTPUT_THROUGHPUT - end
    if gap > 0:
        lines += [
            "",
            f"{last} 离 vLLM 还差 {gap:.1f} tok/s，vLLM 是 {last} 的 "
            f"{VLLM_OUTPUT_THROUGHPUT / end:.2f} 倍。",
        ]
    else:
        lines += ["", f"{last} 比 vLLM 高 {-gap:.1f} tok/s。"]
    return lines


# ---- main ---------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--model-path", required=True, help="bf16 checkpoint 所在的本地目录")
    ap.add_argument("--fp8-model-path", required=True, help="FP8 checkpoint 所在的本地目录")
    ap.add_argument("--repo", default=os.getcwd(), help="sglang-jax 仓库根目录（默认当前目录）")
    ap.add_argument("--out", default="bringup-parity-r1", help="输出目录")
    ap.add_argument("--port", type=int, default=30000, help="服务端口")
    ap.add_argument("--tp", type=int, default=rb.EXPECTED_DEVICES, help="张量并行度（默认 8）")
    ap.add_argument("--start-timeout", type=int, default=7200, help="每次启动最多等多少秒")
    ap.add_argument("--stages", default=",".join(STAGES), help="要跑的阶段，逗号分隔")
    ap.add_argument("--gpqa-seeds", default=s2.GPQA_SEEDS, help="GPQA-Diamond 的 seed，逗号分隔")
    ap.add_argument(
        "--extra-server-args",
        nargs=argparse.REMAINDER,
        default=[],
        help="追加给 launch_server 的参数（放在命令最后）",
    )
    ap.add_argument("--dry-run", action="store_true", help="只打印要执行的命令")
    args = ap.parse_args()
    args.model_path = str(Path(args.model_path).resolve())
    args.fp8_model_path = str(Path(args.fp8_model_path).resolve())
    args.repo = str(Path(args.repo).resolve())
    args.start_timeout = max(args.start_timeout, s2.READY_TIMEOUT_MIN)
    fp8 = fp8_args(args)

    if args.dry_run:
        print("# 代码基底")
        for name, commit in BASES.items():
            parts = "、".join(f"Part {p}" for p, b in PART_BASE.items() if b == name)
            print(f"{name}  {commit}：{parts}")
        launches = (
            ("A1（基底 A，bf16，cookbook §2.3）", args, LAUNCH_COOKBOOK),
            ("B1（基底 B，bf16）", args, LAUNCH),
            ("C3（基底 B，FP8）", fp8, LAUNCH),
            ("D1（基底 D，FP8，开 overlap）", fp8, LAUNCH_OVERLAP),
            ("D5（基底 D，bf16，开 overlap）", args, LAUNCH_OVERLAP),
        )
        for name, a, launch in launches:
            print(f"\n# {name}\n{shlex.join(rb.server_cmd(a, 'qsa_sparse', launch))}")
        print(f"\n# C1\n{shlex.join([sys.executable, *MOE_TUNE_FP8_ARGS])}")
        out = Path(args.out) / PREFILL_DIR
        for batch, input_len in PREFILL_PROFILES:
            print(f"\n# C7 prefill {batch} × {input_len}（停掉服务之后）")
            print(shlex.join(prefill_cmd(fp8, batch, input_len, out)))
        return 0

    # SSH 断开或被 kill 时照 Ctrl-C 处理：先停掉服务、切回原来的分支，再写报告、打包
    signal.signal(signal.SIGHUP, signal.default_int_handler)
    signal.signal(signal.SIGTERM, signal.default_int_handler)
    run = rb.Run(Path(args.out))
    stages = {s.strip().upper() for s in args.stages.split(",")}
    if "C1" in stages and "C2" not in stages:
        stages.add("C2")
        run.log("C1 重跑时 C2 一起跑：数值检查要针对新调出的条目")
    for s in stages - {"C5"}:  # C5 按 seed 落盘，断点重跑时保留没重跑的 seed
        run.results.pop(s, None)
    run.save()
    run.log(f"要跑的阶段：{sorted(stages)}；输出目录：{run.out.resolve()}")
    # 三次启动、bench_one_batch 和调参共用编译缓存
    os.environ.setdefault("JAX_COMPILATION_CACHE_DIR", str((run.out / "jit_cache").resolve()))
    copy_scripts(run)
    original = current_ref(args)

    e0_failed = False
    try:
        if "E0" in stages and not stage_e0(run, args):
            e0_failed = True
            stages = set()
        for part in PARTS:
            run_part(run, args, part, stages)
    finally:
        (run.out / "workspace.diff").write_text(rb.git(args.repo, "diff", "HEAD").stdout)
        if not e0_failed:
            restore(run, args, original)
        run.log(f"报告：{write_report(run, args, rb.git_info(args.repo))}")
        if e0_failed:
            run.log("环境检查（E0）没有通过：按上面列出的问题修好后重跑，这次不打包、不用发回")
        else:
            main_archive, profile_archive = f"{run.out}.tar.gz", f"{run.out}-profile.tar.gz"
            s2.pack(run, main_archive, lambda n: "jit_cache" not in n and "/profile_" not in n)
            s2.pack(run, profile_archive, lambda n: n == run.out.name or "/profile_" in n)
            run.log(f"请把这两个压缩包发回：{main_archive}、{profile_archive}")
    return 0 if all(v["status"] != "FAIL" for v in run.results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
