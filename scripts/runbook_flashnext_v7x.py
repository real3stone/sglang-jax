"""Qwen3.8-Flash-Next 在 TPU v7x 上的跑通测试（N-gram / PLE 层关闭）。

确认模型能加载、能出 token，并验证 QSA 稀疏注意力的实现是否正确。一次跑完所有阶段，
结束时留下一份报告和一个压缩包。

各阶段
  C0  环境检查：8 个 TPU 设备、131 个权重文件、--repo 是 sglang-jax 仓库根目录、sgl_jax 从
      这个仓库导入、代码包含 REQUIRED_COMMIT（这份 runbook 对应的代码版本）。不满足就停。
  C1  MoE kernel 自检 + 稠密注意力（fa）真权重启动。
      先在 8 个设备上跑 fused MoE kernel 在这个模型形状（hidden 2560、intermediate 640、
      top-10）上的数值测试：这组形状查不到调优配置，退到了别处从没执行过的分块，而 C3
      两边用的是同一个 MoE，算错了发现不了。没通过就不启动服务。
      启动后加载摘要要逐字等于 "consumed=1157, skipped=501, missing=0, unexpected=0"；
      再发一个请求，能正常出 token、logprob 里没有 NaN。
  C2  fa 的参考输出：8 个短 prompt 贪心生成，记下每个 token 和它的 logprob；3 个 needle
      （长文开头埋一个 6 位口令，末尾问它；生成 512 个 token，模型先在 <think> 里思考再回答，
      输出里任意位置出现口令就算找到，另外记下口令是否出现在 </think> 之后）；再跑一遍
      bench_serving，参数和 C4 相同，给 C4 做对比。
  C3  稀疏注意力（qsa_sparse）启动 + 逐 token 对照。
      QSA 是这个模型的稀疏注意力：indexer 为每个 query 选出最多 2048 个 token 的 KV 块，
      注意力只在这些块上算。短 prompt 在预算以内，全部块都会被选中，数学上和 fa 相同，
      差别只来自两套 kernel 的 bf16 数值，所以要求前 8 个 token 一致：一半以上的 prompt
      做不到判 FAIL；个别做不到，或者一致部分的概率差超过 0.1，判 WARN。第一个 token
      就分叉指向 prefill，之后才分叉指向 decode。needle 检验稀疏选块能不能把开头的内容
      找回来（第一个在预算以内，另外两个超出预算）：fa 找到了而 qsa 没找到，判 WARN。
      C3 只验证 QSA 这一部分：GDN、MoE、超连接两边共用，它们的错 C3 看不到。
  C4  性能（qsa_sparse）：bench_serving 随机 512 token 输入 / 128 token 输出、100 个请求、
      并发 8，配置和 sgl-project/sglang-jax#1656 一致。记录吞吐、TTFT、TPOT，要求 100 个
      请求全部成功；和 C2 里 fa 的同口径数字放在一起，才拆得出 QSA 自身的开销，缺了 fa
      的数字判 WARN。数字不含 PLE 的开销。C3 的逐 token 对照判 FAIL 时不跑：实现有错，
      性能数字没有意义。

状态
  PASS 通过；WARN 能跑但有可疑之处；FAIL 失败；未运行：这次没选这个阶段，或者前面的
  结果已经说明它没有意义。
  C0 的 FAIL 是本机环境问题：按列出的问题修好后重跑，这种情况不打包、不用发回。
  C1–C4 不管什么结果都不用自己排查，把压缩包原样发回。只有一个例外：C3 或 C4 失败、
  而 server_qsa_sparse.log 里有 SparseCore top-k 的报错（topk_multitile、sc_topk、
  SparseCore）时，运行快结束时日志里会有提示；照提示 export DSA_SC_TOPK=0，再用同一个 --out 跑
  --stages C3,C4，然后一起发回。

服务一共启动两次：C1、C2 用 fa，C3、C4 用 qsa_sparse，每次都要加载约 234 GiB 权重。
某个阶段出了异常（比如请求超时），只要服务还活着，后面的阶段照样跑。

前置条件
  * TPU v7x，一台机器上 4 颗芯片 = 8 个 JAX 设备（拓扑 2x2x1）
  * Python 3.12 或 3.13。sglang-jax 用可编辑方式安装、带 tpu 依赖（会装上 jax[tpu]==0.11.1）：
      git clone -b bringup/qwen4-exp-v7x https://github.com/real3stone/sglang-jax
      cd sglang-jax && pip install -e "python[tpu]"
  * checkpoint：Hugging Face 上的 Qwen/Qwen3.8-Flash-Next（335 GiB，131 个 safetensors
    文件，含 tokenizer），磁盘至少留 400 GB：
      hf download Qwen/Qwen3.8-Flash-Next --local-dir /data/Qwen3.8-Flash-Next
  * 能访问外网：bench_serving 要下载 ShareGPT 数据集

用法：在 sglang-jax 仓库根目录下运行，放在 tmux / screen 里或者用 nohup，免得 SSH 断开
把它带走。
  python scripts/runbook_flashnext_v7x.py --model-path /data/Qwen3.8-Flash-Next --out ./bringup
  python scripts/runbook_flashnext_v7x.py ... --stages C3,C4   # 只重跑 qsa_sparse 那次启动
  python scripts/runbook_flashnext_v7x.py ... --dry-run        # 只打印两条服务启动命令
中途被打断后重跑之前，先确认没有残留的服务进程占着 TPU（pkill -f sgl_jax.launch_server）。

跑完把 <out>.tar.gz 发回来。
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shlex
import shutil
import signal
import subprocess
import sys
import tarfile
import time
import urllib.error
import urllib.request
from pathlib import Path

import psutil

STAGES = ("C0", "C1", "C2", "C3", "C4")
NOT_RUN = "未运行"
STAGE_INFO = {
    "C0": "环境检查：TPU 设备、权重文件、仓库和代码版本",
    "C1": "MoE kernel 自检 + fa 真权重启动：加载摘要逐字比对，能正常出 token",
    "C2": "fa 参考输出：短 prompt 逐 token 结果、needle 长文检索、压测对比数字",
    "C3": "qsa_sparse 启动：逐 token 对照 C2 验证 QSA 实现正确；needle 验证稀疏选块",
    "C4": "性能：bench_serving 512 入 / 128 出、100 个请求、并发 8，对比 fa（不含 PLE）",
}

EXPECTED_DEVICES = 8
EXPECTED_SAFETENSORS = 131
# 这份 runbook 对应的代码版本，C0 要求代码至少包含这个提交
REQUIRED_COMMIT = "5dcf8188"
# consumed = 映射上并加载的张量；skipped = 视觉 333 + MTP 31 + N-gram 137。
EXPECTED_LOAD_SUMMARY = "consumed=1157, skipped=501, missing=0, unexpected=0"
PLE_OFF = {"text_config": {"ple_layer_ids": []}}

MIN_AGREEING_TOKENS = 8  # 每个 prompt 前 8 个 token 要一致，之后 bf16 下接近平局的 token 可能翻转
# 两边选了同一个 token 的位置上，概率最大允许差。用概率不用 logprob：低置信 token 上取对数
# 会把很小的差距放大（p=0.22 对 0.13，logprob 差 0.52）。
MAX_PROB_DIFF_SHARED = 0.1
MAX_NEW_TOKENS = 32
# 循环状态池也要显式给成这个数：只给 --max-running-requests 时，服务端按自动路径再预留约
# 1/4 的快照槽，请求池只剩 12，低于 fused MoE 在 ep=8 下要求的最小值（2 * ep_size = 16）。
MAX_RUNNING_REQUESTS = 16
# fused MoE 在这个模型形状上的数值测试（Python 标准库 unittest 就能跑，不需要 pytest）
MOE_SELFTEST = "sgl_jax.test.kernels.fused_moe_v1_test"
MOE_SELFTEST_FILTER = "flash_next_fallback_tiles"
# 服务端第一次遇到没预编译过的请求形状时要现场编译整个模型，这个模型在 v7x 上编译一次要多久
# 没有实测过。服务端 watchdog（一个 batch 卡住多久就让服务自杀）和客户端请求超时都放宽到这个数。
COMPILE_TIMEOUT = 1800

BENCH = {"input_len": 512, "output_len": 128, "num_prompts": 100, "concurrency": 8}

PROMPTS = [
    "The capital of France is",
    "def fibonacci(n):\n    ",
    "Water boils at 100 degrees Celsius at sea level. At higher altitudes,",
    "Translate to French: 'The weather is nice today.'\nFrench:",
    "List three prime numbers greater than 10:",
    "In 1969, the Apollo 11 mission",
    "The derivative of x^3 with respect to x is",
    "Once upon a time, in a small village by the sea,",
]
# (大约的 prompt token 数, 口令)。第一个在 indexer 预算（2048 token）以内，另外两个超出。
NEEDLES = [(1500, "482917"), (6000, "730164"), (14000, "259803")]
TOKENS_PER_WORD = 2.0  # 填充句大多是数字，Qwen 的分词器把数字逐位切开
NEEDLE_MAX_NEW_TOKENS = 512  # 模型先在 <think> 里思考再回答，要给够 token 让它把口令说出来

PROBE = (
    "import jax, flax, json, sgl_jax, os; d = jax.devices();"
    "print(json.dumps({'jax': jax.__version__, 'flax': flax.__version__, 'n': len(d),"
    "'platform': d[0].platform, 'kind': d[0].device_kind,"
    "'sgl_jax': os.path.dirname(sgl_jax.__file__)}))"
)


class Run:
    def __init__(self, out: Path):
        self.out = out
        self.out.mkdir(parents=True, exist_ok=True)
        self.results: dict[str, dict] = {}
        if (out / "results.json").exists():  # 只重跑部分阶段时，保留其余阶段的结果
            self.results = json.loads((out / "results.json").read_text())

    def log(self, msg: str) -> None:
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        try:
            print(line, flush=True)
        except OSError:  # 终端断开后照样写日志文件
            pass
        with open(self.out / "run.log", "a") as f:
            f.write(line + "\n")

    def record(self, stage: str, status: str, **details) -> None:
        self.results[stage] = {"status": status, **details}
        self.save()
        self.log(f"{stage} 结果：{status}")

    def save(self) -> None:
        text = json.dumps(self.results, indent=2, ensure_ascii=False)
        (self.out / "results.json").write_text(text)


def git(repo: str, *a: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", repo, *a], capture_output=True, text=True, check=False)


def git_info(repo: str) -> dict:
    return {
        "branch": git(repo, "rev-parse", "--abbrev-ref", "HEAD").stdout.strip(),
        "head": git(repo, "rev-parse", "--short", "HEAD").stdout.strip(),
        "dirty": bool(git(repo, "status", "--porcelain", "--untracked-files=no").stdout.strip()),
    }


# server_cmd 的默认启动参数；launch 里给出的项覆盖它们
DEFAULT_LAUNCH = {
    "ple": False,
    "context_length": 32768,
    "max_running_requests": MAX_RUNNING_REQUESTS,
    "precompile_bs_paddings": (16,),
    "precompile_token_paddings": (16, 512, 1024),
}


def server_cmd(args, backend: str, launch: dict | None = None) -> list[str]:
    # 预编译档位、跳过 warmup、随机种子照搬仓库的 Qwen3.5 TPU 测试；超过最大档位的
    # batch 会落到自动补上的 max_padded_num_tokens 那一档，不会报错。
    cfg = {**DEFAULT_LAUNCH, **(launch or {})}
    ple = [] if cfg["ple"] else ["--json-model-override-args", json.dumps(PLE_OFF)]
    running = str(cfg["max_running_requests"])
    return [
        sys.executable, "-u", "-m", "sgl_jax.launch_server",
        "--model-path", args.model_path,
        "--device", "tpu",
        "--dtype", "bfloat16",
        "--tp-size", str(args.tp),
        "--ep-size", str(args.tp),
        "--attention-backend", backend,
        *ple,
        "--context-length", str(cfg["context_length"]),
        "--page-size", "64",
        "--chunked-prefill-size", "2048",
        "--mem-fraction-static", "0.8",
        "--max-running-requests", running,
        "--max-recurrent-state-size", running,
        "--disable-radix-cache",
        "--disable-overlap-schedule",
        "--skip-server-warmup",
        "--random-seed", "3",
        "--precompile-bs-paddings", *map(str, cfg["precompile_bs_paddings"]),
        "--precompile-token-paddings", *map(str, cfg["precompile_token_paddings"]),
        "--watchdog-timeout", str(COMPILE_TIMEOUT),
        "--host", "127.0.0.1",
        "--port", str(args.port),
    ] + args.extra_server_args  # fmt: skip


class Server:
    def __init__(self, run: Run, args, backend: str, launch: dict | None = None, label: str = ""):
        self.run, self.args, self.backend, self.launch = run, args, backend, launch
        # 同一个后端启动两次时，label 把日志、压测和 profile 的文件名分开
        self.label = label or backend
        self.base = f"http://127.0.0.1:{args.port}"
        self.log_path = run.out / f"server_{self.label}.log"
        self.proc: subprocess.Popen | None = None
        self.children: list[psutil.Process] = []

    def __enter__(self):
        cmd = server_cmd(self.args, self.backend, self.launch)
        env = dict(os.environ)
        # 两次启动共用编译缓存，第二次能省掉一部分编译时间
        env.setdefault("JAX_COMPILATION_CACHE_DIR", str(self.run.out / "jit_cache"))
        self.run.log(f"启动服务（{self.backend}）：{shlex.join(cmd)}")
        if self.log_path.exists():  # 只重跑部分阶段时，留下上一次启动的日志
            n = 1
            while self.log_path.with_suffix(f".{n}.log").exists():
                n += 1
            self.log_path.rename(self.log_path.with_suffix(f".{n}.log"))
        self.proc = subprocess.Popen(
            cmd,
            cwd=self.args.repo,
            env=env,
            stdout=open(self.log_path, "w"),
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        return self

    def dead_reason(self) -> str | None:
        """服务还在跑时返回 None，否则返回原因。"""
        if self.proc.poll() is not None:
            return f"服务进程已退出（返回码 {self.proc.returncode}），请查看 {self.log_path}"
        for child in self.children:
            try:
                if child.is_running() and child.status() != psutil.STATUS_ZOMBIE:
                    continue
            except psutil.NoSuchProcess:
                pass
            return f"服务的子进程 {child.pid} 已退出，请查看 {self.log_path}"
        return None

    def alive(self) -> bool:
        return self.dead_reason() is None

    def wait_ready(self) -> float:
        t0 = time.time()
        while time.time() - t0 < self.args.start_timeout:
            if reason := self.dead_reason():
                raise RuntimeError(reason)
            try:
                urllib.request.urlopen(f"{self.base}/health", timeout=10)
                # scheduler、detokenizer 子进程被信号杀掉时，launch_server 只打一行警告、
                # 自己不退出，所以要连同子进程一起看
                self.children = psutil.Process(self.proc.pid).children()
                return time.time() - t0
            except (urllib.error.URLError, ConnectionError, TimeoutError):
                time.sleep(15)
        raise RuntimeError(f"等了 {self.args.start_timeout} 秒服务仍未就绪，请查看 {self.log_path}")

    def generate(self, text: str, max_new_tokens: int) -> dict:
        payload = {
            "text": text,
            "sampling_params": {"temperature": 0, "max_new_tokens": max_new_tokens},
            "return_logprob": True,
        }
        req = urllib.request.Request(
            f"{self.base}/generate",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=COMPILE_TIMEOUT) as r:
            out = json.loads(r.read())
        lps = out["meta_info"]["output_token_logprobs"]
        return {
            "text": out["text"],
            "token_ids": [int(t[1]) for t in lps],
            "logprobs": [float(t[0]) for t in lps],
            "prompt_tokens": out["meta_info"].get("prompt_tokens"),
        }

    def __exit__(self, *exc):
        if self.proc and self.proc.poll() is None:
            self.run.log(f"停止服务（{self.backend}）")
            os.killpg(self.proc.pid, signal.SIGTERM)
            try:
                self.proc.wait(timeout=180)
            except subprocess.TimeoutExpired:
                os.killpg(self.proc.pid, signal.SIGKILL)
                self.proc.wait()
        time.sleep(30)  # 等 TPU 运行时释放芯片，再进行下一次启动


def stage_launch(
    run: Run, args, backend: str, steps, launch: dict | None = None, label: str = ""
) -> None:
    """一次服务启动依次跑 ``steps`` = [(阶段, fn(srv, ready, checks))]。

    某个阶段出了异常只记在它自己身上，服务还活着就接着跑下一个；服务退出了或者
    根本没起来，剩下的阶段都记成 FAIL。
    """
    pending = [stage for stage, _ in steps]
    try:
        with Server(run, args, backend, launch, label) as srv:
            ready = srv.wait_ready()
            summary = [
                line.rstrip() for line in open(srv.log_path) if "WeightLoader summary" in line
            ]
            checks = {
                "load_summary": summary[-1] if summary else None,
                "load_summary_ok": bool(summary) and EXPECTED_LOAD_SUMMARY in summary[-1],
            }
            run.log(
                f"服务就绪（{backend}），用时 {ready / 60:.1f} 分钟；加载摘要：{checks['load_summary']}"
            )
            for stage, fn in steps:
                pending.remove(stage)
                run.log(f"===== {stage}：{STAGE_INFO[stage]}")
                try:
                    fn(srv, ready, checks)
                except Exception as e:  # noqa: BLE001 - 记在这个阶段上，接着跑下一个
                    run.log(f"{stage} 出错：{type(e).__name__}: {e}")
                    run.record(stage, "FAIL", error=f"{type(e).__name__}: {e}")
                if reason := srv.dead_reason():
                    break
    except Exception as e:  # noqa: BLE001 - 这次启动失败，也要留下报告
        reason = f"{type(e).__name__}: {e}"
        run.log(f"{backend} 这次启动出错：{reason}")
    for stage in pending:
        run.record(stage, "FAIL", error=f"没有运行：{reason}")


# ---- C0 -----------------------------------------------------------------------


def host_mem_available_gib() -> float | None:
    meminfo = Path("/proc/meminfo")
    if not meminfo.exists():
        return None
    for line in meminfo.read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / 2**20
    return None


def stage_c0(run: Run, args, stage: str = "C0", min_mem_gib: float | None = None) -> bool:
    run.log(f"===== {stage}：{STAGE_INFO[stage]}")
    problems = []
    mem_gib = host_mem_available_gib()
    if min_mem_gib is not None and (mem_gib is None or mem_gib < min_mem_gib):
        have = "读不到" if mem_gib is None else f"{mem_gib:.0f} GiB"
        problems.append(
            f"主机可用内存 {have}，至少要 {min_mem_gib:.0f} GiB："
            "n-gram 表（95.4 GiB）常驻主机内存，还要留出余量"
        )
    pkg = Path(args.repo, "python", "sgl_jax")
    if not pkg.is_dir():
        problems.append(
            f"--repo（默认是当前目录）不是 sglang-jax 仓库根目录：{args.repo}。"
            "请 cd 进仓库根目录再运行，或者用 --repo 指定"
        )

    # JAX 放在子进程里探测：驱动脚本自己若初始化了 TPU，后面起的服务就拿不到设备
    r = subprocess.run(
        [sys.executable, "-c", PROBE], capture_output=True, text=True, timeout=600, check=False
    )
    out = r.stdout.strip().splitlines()
    if r.returncode != 0 or not out:
        error = " | ".join(r.stderr.strip().splitlines()[-5:]) or f"返回码 {r.returncode}"
        info = {"error": error}
        problems.append(f"环境探测失败（import jax / flax / sgl_jax 或取 TPU 设备时出错）：{error}")
    else:
        info = json.loads(out[-1])
        install = '在仓库根目录执行 pip install -e "python[tpu]"'
        if info["platform"] != "tpu":
            problems.append(f"JAX 平台是 {info['platform']}，不是 tpu：{install}")
        if info["n"] != args.tp:
            problems.append(
                f"期望 {args.tp} 个 JAX 设备，实际 {info['n']} 个；"
                "如果之前的运行被打断过，先确认没有残留的服务进程占着 TPU"
            )
        if pkg.is_dir() and Path(info["sgl_jax"]).resolve() != pkg.resolve():
            problems.append(
                f"sgl_jax 是从 {info['sgl_jax']} 导入的，不是 --repo 指向的仓库：{install}"
            )

    model = Path(args.model_path)
    n_st = len(list(model.glob("*.safetensors")))
    if not model.is_dir():
        problems.append(f"--model-path 不是已有的目录：{model}")
    elif not (model / "config.json").exists():
        problems.append("--model-path 目录下没有 config.json")
    if n_st != EXPECTED_SAFETENSORS:
        problems.append(f"期望 {EXPECTED_SAFETENSORS} 个 safetensors 文件，实际 {n_st} 个")

    repo = git_info(args.repo) if pkg.is_dir() else {}
    if pkg.is_dir():
        # 0 = 包含；1 = 不包含；其他（比如 128）= 本地根本没有这个提交，也就是没 fetch 过
        ancestor = git(args.repo, "merge-base", "--is-ancestor", REQUIRED_COMMIT, "HEAD")
        if ancestor.returncode != 0:
            problems.append(
                f"代码太旧（HEAD={repo['head']}），需要包含提交 {REQUIRED_COMMIT}。请在仓库根目录执行："
                " git fetch origin && git reset --hard origin/bringup/qwen4-exp-v7x"
            )

    for p in problems:
        run.log(f"  问题：{p}")
    run.record(
        stage,
        "FAIL" if problems else "PASS",
        problems=problems,
        devices=info,
        mem_available_gib=None if mem_gib is None else round(mem_gib, 1),
        safetensors_files=n_st,
        disk_free=f"{shutil.disk_usage(model).free / 2**30:.0f} GiB" if model.is_dir() else None,
        repo=repo,
    )
    return not problems


# ---- C1 / C2：fa ----------------------------------------------------------------


def run_prompts(srv: Server) -> list[dict]:
    return [{"prompt": p, **srv.generate(p, MAX_NEW_TOKENS)} for p in PROMPTS]


def needle_prompt(approx_tokens: int, code: str) -> str:
    head = f"Remember this: the secret passphrase is {code}.\n\n"
    tail = "\n\nWhat is the secret passphrase mentioned at the very beginning? The secret passphrase is"
    lines, i = [], 0
    while len(" ".join(lines).split()) * TOKENS_PER_WORD < approx_tokens:
        i += 1
        lines.append(
            f"Record {i}: the survey of district {i % 97} counted {i * 7 % 1000} trees, "
            f"{i * 13 % 50} wells and {i * 3 % 20} bridges."
        )
    return head + "\n".join(lines) + tail


def run_needles(run: Run, srv: Server, needles=NEEDLES) -> list[dict]:
    out = []
    for n, code in needles:
        g = srv.generate(needle_prompt(n, code), NEEDLE_MAX_NEW_TOKENS)
        text = g["text"]
        answer = text.split("</think>", 1)[1] if "</think>" in text else ""
        entry = {
            "target_tokens": n,
            "prompt_tokens": g["prompt_tokens"],
            "found": code in text,
            "after_think": code in answer,
            "output": text,
            "token_ids": g["token_ids"],
        }
        run.log(f"  needle 约 {n} token（实际 {g['prompt_tokens']}）：{needle_status(entry)}")
        out.append(entry)
    return out


def needle_status(needle: dict | None) -> str:
    if needle is None:
        return "-"
    if not needle["found"]:
        return "没找到"
    return "找到（</think> 之后）" if needle["after_think"] else "找到（只在思考里）"


def moe_selftest(run: Run, args) -> str | None:
    """跑 fused MoE kernel 在这个模型形状上的数值测试，没通过时返回原因。"""
    log = run.out / "moe_selftest.log"
    cmd = [sys.executable, "-m", "unittest", "-k", MOE_SELFTEST_FILTER, MOE_SELFTEST]
    run.log(f"  MoE kernel 自检：{shlex.join(cmd)}")
    with open(log, "w") as f:
        r = subprocess.run(cmd, cwd=args.repo, stdout=f, stderr=subprocess.STDOUT, check=False)
    tail = log.read_text(errors="replace").strip().splitlines()[-1:] or [""]
    run.log(f"  MoE kernel 自检：返回码 {r.returncode}，{tail[0]}")
    if r.returncode != 0 or "Ran 0 tests" in log.read_text(errors="replace"):
        return f"MoE kernel 自检没通过（返回码 {r.returncode}），请查看 {log}"
    return None


def stage_fa(run: Run, args) -> None:
    run.log("===== C1：MoE kernel 自检（通过后再启动服务）")
    error = moe_selftest(run, args)
    if error:
        run.record("C1", "FAIL", error=error)
        run.record("C2", NOT_RUN, reason="C1 没有通过")
        return

    def c1(srv, ready, checks):
        smoke = srv.generate(PROMPTS[0], 16)
        bad = not smoke["text"].strip() or any(math.isnan(x) for x in smoke["logprobs"])
        run.log(f"  冒烟请求：{PROMPTS[0]!r} -> {smoke['text']!r}")
        if not checks["load_summary_ok"]:
            run.log(f"  加载摘要和期望不一致（期望 {EXPECTED_LOAD_SUMMARY}）")
        if bad:
            run.log("  冒烟请求输出为空或 logprob 里有 NaN")
        ok = checks["load_summary_ok"] and not bad
        run.record(
            "C1",
            "PASS" if ok else "FAIL",
            ready_minutes=round(ready / 60, 1),
            smoke=smoke["text"],
            **checks,
        )

    def c2(srv, ready, checks):
        if run.results["C1"]["status"] != "PASS":
            run.record("C2", NOT_RUN, reason="C1 没有通过")
            return
        prompts = run_prompts(srv)
        (run.out / "c2_prompts.json").write_text(json.dumps(prompts, indent=2, ensure_ascii=False))
        needles = run_needles(run, srv)
        try:
            bench = run_bench(run, srv, args)
        except Exception as e:  # noqa: BLE001 - 压测只给 C4 做对比，失败不影响参考输出
            bench = {"error": f"{type(e).__name__}: {e}"}
        run.log(f"  bench_serving（fa）：{bench}")
        run.record("C2", "PASS", needles=needles, bench=bench)

    stage_launch(run, args, "fa", [("C1", c1), ("C2", c2)])


# ---- C3 / C4：qsa_sparse ---------------------------------------------------------


def compare(ref: list[dict], got: list[dict]) -> dict:
    """fa 与 qsa_sparse 逐 token 对照。

    一边先结束（输出了 EOS）、另一边还在生成，算在短的那一边结束的位置分叉；
    两边完全相同的输出，不管多短都不算分叉。
    """
    rows, first_token_flips, early = [], 0, 0
    for a, b in zip(ref, got):
        n = min(len(a["token_ids"]), len(b["token_ids"]))
        div = next((i for i in range(n) if a["token_ids"][i] != b["token_ids"][i]), None)
        if div is None and len(a["token_ids"]) != len(b["token_ids"]):
            div = n
        shared = n if div is None else div
        d = [
            abs(math.exp(x) - math.exp(y))
            for x, y in zip(a["logprobs"][:shared], b["logprobs"][:shared])
        ]
        first_token_flips += div == 0
        early += div is not None and div < MIN_AGREEING_TOKENS
        rows.append(
            {
                "prompt": a["prompt"][:40],
                "first_divergence": div,
                "max_prob_diff_shared": max(d) if d else None,
                "fa": a["text"][:80],
                "qsa": b["text"][:80],
            }
        )
    worst = max((r["max_prob_diff_shared"] or 0) for r in rows)
    if early > len(rows) // 2:
        status = "FAIL"
    elif early or worst > MAX_PROB_DIFF_SHARED:
        status = "WARN"
    else:
        status = "PASS"
    return {
        "status": status,
        "first_token_flips": first_token_flips,
        "early_divergences": early,
        "worst_shared_prob_diff": worst,
        "rows": rows,
    }


def run_logged(
    args, cmd, log: Path, srv: Server | None = None, timeout: float | None = None
) -> int:
    """在仓库根目录跑一条命令，输出写进 log，返回返回码。

    srv 给出时每分钟看一次服务，服务退出就杀掉命令并抛异常；timeout 到了也杀掉，返回 -9。
    """
    deadline = None if timeout is None else time.time() + timeout
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}  # 命令被杀掉时，已有的输出也留在日志里
    with open(log, "w") as f:
        proc = subprocess.Popen(
            cmd, cwd=args.repo, env=env, stdout=f, stderr=subprocess.STDOUT, start_new_session=True
        )
        try:
            while True:
                try:
                    return proc.wait(timeout=60)
                except subprocess.TimeoutExpired:
                    if srv is not None and (reason := srv.dead_reason()):
                        raise RuntimeError(reason) from None
                    if deadline is not None and time.time() > deadline:
                        return -9
        finally:
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()


def run_bench(run: Run, srv: Server, args, bench: dict = BENCH, tag: str = "") -> dict:
    out_file = run.out / f"bench_serving_{srv.label}{tag}.jsonl"
    out_file.unlink(missing_ok=True)
    cmd = [
        sys.executable, "-m", "sgl_jax.bench_serving",
        "--backend", "sgl-jax",
        "--base-url", srv.base,
        "--tokenizer", args.model_path,
        "--dataset-name", "random",
        "--random-input-len", str(bench["input_len"]),
        "--random-output-len", str(bench["output_len"]),
        "--random-range-ratio", "1",
        "--num-prompts", str(bench["num_prompts"]),
        "--max-concurrency", str(bench["concurrency"]),
        "--warmup-requests", "0",
        "--output-file", str(out_file),
    ]  # fmt: skip
    run.log(f"  bench_serving：{shlex.join(cmd)}")
    log = run.out / f"bench_serving_{srv.label}{tag}.log"
    rc = run_logged(args, cmd, log, srv=srv)
    if rc != 0 or not out_file.exists():
        raise RuntimeError(f"bench_serving 失败（返回码 {rc}），请查看 {log}")
    result = json.loads(out_file.read_text().strip().splitlines()[-1])
    keep = (
        "completed", "request_throughput", "input_throughput", "output_throughput",
        "total_throughput", "median_ttft_ms", "p99_ttft_ms", "median_tpot_ms", "p99_tpot_ms",
        "median_e2e_latency_ms",
    )  # fmt: skip
    return {k: result[k] for k in keep if k in result}


def stage_qsa(run: Run, args, wanted: set[str]) -> None:
    ref_path = run.out / "c2_prompts.json"

    def c3(srv, ready, checks):
        if not ref_path.exists():
            raise RuntimeError("没有 C2 的参考输出，请先跑 C1、C2")
        got = run_prompts(srv)
        (run.out / "c3_prompts.json").write_text(json.dumps(got, indent=2, ensure_ascii=False))
        cmp = compare(json.loads(ref_path.read_text()), got)
        run.log(
            f"  逐 token 对照：{cmp['status']}；第一个 token 就分叉 {cmp['first_token_flips']} 个，"
            f"前 {MIN_AGREEING_TOKENS} 个 token 内分叉 {cmp['early_divergences']} 个，"
            f"一致部分概率最大差 {cmp['worst_shared_prob_diff']:.4f}"
        )
        details = {"compare": cmp, "load_summary_ok": checks["load_summary_ok"]}
        try:
            needles = run_needles(run, srv)
        except Exception as e:  # noqa: BLE001 - needle 出错时，逐 token 对照的结论照样留下
            run.record("C3", "FAIL", error=f"needle 出错：{type(e).__name__}: {e}", **details)
            return
        dense = {
            n["target_tokens"]: n["found"] for n in run.results.get("C2", {}).get("needles", [])
        }
        lost = [
            n["target_tokens"] for n in needles if dense.get(n["target_tokens"]) and not n["found"]
        ]
        if lost:
            run.log(f"  fa 找到了、qsa 没找到的 needle：{lost}（稀疏选块可疑）")
        status = cmp["status"]
        if status == "PASS" and (lost or not checks["load_summary_ok"]):
            status = "WARN"
        run.record("C3", status, needles=needles, needles_lost=lost, **details)

    def c4(srv, ready, checks):
        if run.results.get("C3", {}).get("compare", {}).get("status") == "FAIL":
            run.record("C4", NOT_RUN, reason="C3 的逐 token 对照没通过，QSA 实现大概率有错")
            return
        bench = run_bench(run, srv, args)
        run.log(f"  bench_serving（qsa_sparse）：{bench}")
        status = "PASS" if bench.get("completed") == BENCH["num_prompts"] else "FAIL"
        bench_fa = run.results.get("C2", {}).get("bench") or {}
        if status == "PASS" and "completed" not in bench_fa:
            run.log("  没有 fa 的压测数字，拆不出 QSA 自身的开销")
            status = "WARN"
        run.record("C4", status, bench=bench, bench_fa=bench_fa)

    steps = [(s, fn) for s, fn in (("C3", c3), ("C4", c4)) if s in wanted]
    stage_launch(run, args, "qsa_sparse", steps)
    failed = any(run.results.get(s, {}).get("status") == "FAIL" for s, _ in steps)
    log = run.out / "server_qsa_sparse.log"
    if failed and os.environ.get("DSA_SC_TOPK") != "0" and log.exists():
        text = log.read_text(errors="replace")
        if any(k in text for k in ("topk_multitile", "sc_topk", "SparseCore")):
            run.log(
                "server_qsa_sparse.log 里有 SparseCore top-k 相关的报错：先 export DSA_SC_TOPK=0，"
                "再用同一个 --out 跑 --stages C3,C4，然后把压缩包一起发回"
            )


# ---- 报告 ---------------------------------------------------------------------


def write_report(run: Run, repo: dict) -> Path:
    r = run.results
    dirty = "（有未提交的改动）" if repo.get("dirty") else ""
    lines = [
        "# Qwen3.8-Flash-Next 跑通测试（PLE 关闭）",
        "",
        f"代码：{repo.get('branch')} @ {repo.get('head')}{dirty}",
        "",
        "| 阶段 | 结果 | 测什么 |",
        "| --- | --- | --- |",
    ]
    lines += [f"| {s} | {r.get(s, {}).get('status', NOT_RUN)} | {STAGE_INFO[s]} |" for s in STAGES]
    lines += [""]
    for s in STAGES:
        c = r.get(s, {})
        lines += [f"- {s} 问题：{p}" for p in c.get("problems", [])]
        lines += [f"- {s}：{c[k]}" for k in ("error", "reason") if k in c]
    if "C1" in r and "load_summary" in r["C1"]:
        lines += ["", f"加载摘要：`{r['C1']['load_summary']}`（期望 `{EXPECTED_LOAD_SUMMARY}`）"]
    cmp = r.get("C3", {}).get("compare")
    if cmp:
        lines += [
            "",
            f"逐 token 对照：第一个 token 就分叉 {cmp['first_token_flips']} 个，前 {MIN_AGREEING_TOKENS} 个 "
            f"token 内分叉 {cmp['early_divergences']} 个，一致部分概率最大差 "
            f"{cmp['worst_shared_prob_diff']:.4f}",
            "",
            "| prompt | 首次分叉位置 | 一致部分概率最大差 |",
            "| --- | --- | --- |",
        ]
        for x in cmp["rows"]:
            div = "无" if x["first_divergence"] is None else x["first_divergence"]
            d = "-" if x["max_prob_diff_shared"] is None else f"{x['max_prob_diff_shared']:.3f}"
            lines += [f"| {x['prompt']!r} | {div} | {d} |"]
    fa, qsa = (
        {n["target_tokens"]: n for n in r.get(s, {}).get("needles", [])} for s in ("C2", "C3")
    )
    if fa or qsa:
        lines += ["", "| needle | 实际 token 数 | fa | qsa |", "| --- | --- | --- | --- |"]
        for n, _ in NEEDLES:
            a, b = fa.get(n), qsa.get(n)
            tokens = (a or b or {}).get("prompt_tokens", "-")
            lines += [f"| {n} | {tokens} | {needle_status(a)} | {needle_status(b)} |"]
    lines += bench_table(r.get("C2", {}).get("bench"), r.get("C4", {}).get("bench"))
    path = run.out / "report.md"
    path.write_text("\n".join(lines) + "\n")
    return path


def bench_table(fa: dict | None, qsa: dict | None) -> list[str]:
    if not fa and not qsa:
        return []
    fa, qsa = fa or {}, qsa or {}
    lines = [
        "",
        f"性能（{BENCH}，不含 PLE 的开销）",
        "",
        "| 指标 | fa | qsa_sparse |",
        "| --- | --- | --- |",
    ]
    for k in dict.fromkeys([*fa, *qsa]):
        lines += [f"| {k} | {fa.get(k, '-')} | {qsa.get(k, '-')} |"]
    return lines


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--model-path", required=True, help="完整 checkpoint 所在的本地目录")
    ap.add_argument("--repo", default=os.getcwd(), help="sglang-jax 仓库根目录（默认当前目录）")
    ap.add_argument("--out", default="bringup", help="输出目录")
    ap.add_argument("--port", type=int, default=30000, help="服务端口")
    ap.add_argument("--tp", type=int, default=EXPECTED_DEVICES, help="张量并行度（默认 8）")
    ap.add_argument("--start-timeout", type=int, default=7200, help="每次启动最多等多少秒")
    ap.add_argument("--stages", default=",".join(STAGES), help="要跑的阶段，逗号分隔")
    ap.add_argument(
        "--extra-server-args",
        nargs=argparse.REMAINDER,
        default=[],
        help="追加给 launch_server 的参数（放在命令最后）",
    )
    ap.add_argument("--dry-run", action="store_true", help="只打印两条服务启动命令")
    args = ap.parse_args()
    args.model_path = str(Path(args.model_path).resolve())
    args.repo = str(Path(args.repo).resolve())

    if args.dry_run:
        for backend in ("fa", "qsa_sparse"):
            print(shlex.join(server_cmd(args, backend)), end="\n\n")
        return 0

    # SSH 断开或被 kill 时照 Ctrl-C 处理：先停掉服务，再写报告、打包
    signal.signal(signal.SIGHUP, signal.default_int_handler)
    signal.signal(signal.SIGTERM, signal.default_int_handler)
    run = Run(Path(args.out))
    stages = {s.strip().upper() for s in args.stages.split(",")}
    # fa 那次启动总是 C1、C2 一起跑；重跑它时，C3、C4 对照的是旧的参考输出，一起清掉
    stale = stages | ({"C1", "C2", "C3", "C4"} if stages & {"C1", "C2"} else set())
    for s in stale:
        run.results.pop(s, None)
    if stages & {"C1", "C2"}:
        (run.out / "c2_prompts.json").unlink(missing_ok=True)
    run.save()
    run.log(f"要跑的阶段：{sorted(stages)}；输出目录：{run.out.resolve()}")

    c0_failed = False
    try:
        if "C0" in stages and not stage_c0(run, args):
            c0_failed = True
            stages = set()
        if stages & {"C1", "C2"}:
            stage_fa(run, args)
        if stages & {"C3", "C4"}:
            # 稠密那次都没通过，就不值得再花一次加载时间去起稀疏那次
            if run.results.get("C1", {}).get("status") == "PASS":
                stage_qsa(run, args, stages)
            else:
                run.log("跳过 C3、C4：C1 没有通过")
    finally:
        run.log(f"报告：{write_report(run, git_info(args.repo))}")
        if c0_failed:
            run.log("环境检查（C0）没有通过：按上面列出的问题修好后重跑，这次不打包、不用发回")
        else:
            archive = f"{run.out}.tar.gz"
            with tarfile.open(archive, "w:gz") as tar:
                tar.add(
                    run.out,
                    arcname=run.out.name,
                    filter=lambda t: None if "jit_cache" in t.name else t,
                )
            run.log(f"请把这个压缩包发回：{archive}")
    return 0 if all(v["status"] != "FAIL" for v in run.results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
