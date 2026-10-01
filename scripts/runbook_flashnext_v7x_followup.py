"""Qwen3.8-Flash-Next 跑通测试的补充测试：超出 indexer 预算后，稀疏选块能不能把开头的内容找回来；
以及 QSA 的开销随上下文长度怎么变、时间花在哪些算子上。

和 runbook_flashnext_v7x.py 用同一套启动参数（N-gram / PLE 层关闭），只测那份 runbook 没覆盖到
的部分。结束时留下一份报告和一个压缩包。

各阶段
  F0  环境检查：8 个 TPU 设备、--repo 是 sglang-jax 仓库根目录、代码包含 REQUIRED_COMMIT。
      不满足就停。
  F1  稠密注意力（fa）启动：
      - needle 四档：长文开头埋一个 6 位口令，末尾问它，约 500 / 1500 / 6000 / 14000 token。
        前两档在 indexer 预算（2048 token）以内，后两档超出。贪心生成 512 个 token（模型先在
        <think> 里思考再回答），记下完整输出、有没有找到口令、口令是否出现在 </think> 之后。
      - 性能，两个后端都做，只记录、不判定：
        bench_serving 两组，输入 512 / 输出 128 / 100 个请求，和输入 8192 / 输出 128 / 32 个
        请求，都是并发 8。看 TTFT、TPOT 随输入长度怎么变。
        压测完服务已经预热好，再用 jax.profiler 抓一段 profile：prefill、decode 各几步，
        用 8 个并发请求（输入 512 / 输出 64）凑出来。trace 存在 profile_<后端>/，体积较大。
  F2  稀疏注意力（qsa_sparse）启动：同样四档 needle，同样的性能测量。

判定
  预算内两档：两边数学上相同，都应该找到。
    fa 没找到：F1 判 WARN，问题在测试或模型本身，不算到 QSA 头上；
    fa 找到而 qsa_sparse 没找到：F2 判 FAIL。
  超出预算两档：
    qsa_sparse 找到：PASS；
    qsa_sparse 没找到而 fa 找到：WARN（选块可疑）；
    两边都没找到：WARN（下不了结论）。
    超出预算后 fa 算的已经不是模型训练时的注意力，它只作对比，不当参照。

状态
  PASS 通过；WARN 有可疑之处；FAIL 失败；未运行：这次没选这个阶段。
  F0 的 FAIL 是本机环境问题：按列出的问题修好后重跑，这种情况不打包、不用发回。
  F1、F2 不管什么结果都不用自己排查，把压缩包原样发回（里面有 profile trace，体积较大，
  整个发回）。性能测量失败只记 WARN，不影响 needle 的判定。

用法：和 runbook 一样，在 sglang-jax 仓库根目录下、tmux 里运行。
  python scripts/runbook_flashnext_v7x_followup.py --model-path /data/Qwen3.8-Flash-Next
  python scripts/runbook_flashnext_v7x_followup.py ... --stages F2   # 只重跑 qsa_sparse 那次启动
跑完把 bringup-followup.tar.gz 发回来。
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import signal
import subprocess
import sys
import tarfile
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import runbook_flashnext_v7x as rb  # noqa: E402

STAGES = ("F0", "F1", "F2")
STAGE_INFO = {
    "F0": "环境检查：TPU 设备、仓库和代码版本",
    "F1": "fa：四档 needle（预算内两档必须找到）+ 两种输入长度的压测 + profile",
    "F2": "qsa_sparse：四档 needle（超出预算后选块能不能找回口令）+ 压测 + profile",
}
rb.STAGE_INFO.update(STAGE_INFO)  # stage_launch 打印阶段标题时查这张表

INDEXER_BUDGET = 2048
NEEDLES = [(500, "615208"), *rb.NEEDLES]  # 500 档确认关掉 PLE 后模型本身能找回口令

RANK = {"PASS": 0, "WARN": 1, "FAIL": 2}

# 两种输入长度：看 TPOT 随上下文怎么变（第一组就是 runbook 的 C4）
BENCH_SWEEP = [
    rb.BENCH,
    {"input_len": 8192, "output_len": 128, "num_prompts": 32, "concurrency": 8},
]
PROFILE_LOAD = {"input_len": 512, "output_len": 64, "num_prompts": 8, "concurrency": 8}
PROFILE_REQUEST = {
    "num_steps": 5,
    "profile_by_stage": True,
    "profile_stages": ["prefill", "decode"],
    "host_tracer_level": 2,
    "python_tracer_level": 0,
}
PROFILE_WAIT_SECONDS = 600
PROFILE_MAX_MB = 500  # 超过就只留 xplane.pb


def stage_f0(run: rb.Run, args) -> bool:
    run.log(f"===== F0：{STAGE_INFO['F0']}")
    problems = []
    if not Path(args.repo, "python", "sgl_jax").is_dir():
        problems.append(
            f"--repo（默认是当前目录）不是 sglang-jax 仓库根目录：{args.repo}。"
            "请 cd 进仓库根目录再运行，或者用 --repo 指定"
        )
    else:
        ancestor = rb.git(args.repo, "merge-base", "--is-ancestor", rb.REQUIRED_COMMIT, "HEAD")
        if ancestor.returncode != 0:
            problems.append(
                f"代码太旧，需要包含提交 {rb.REQUIRED_COMMIT}。请在仓库根目录执行："
                " git fetch origin && git reset --hard origin/bringup/qwen4-exp-v7x"
            )
    r = subprocess.run(
        [sys.executable, "-c", rb.PROBE], capture_output=True, text=True, timeout=600, check=False
    )
    out = r.stdout.strip().splitlines()
    if r.returncode != 0 or not out:
        info = {"error": " | ".join(r.stderr.strip().splitlines()[-5:]) or f"返回码 {r.returncode}"}
        problems.append(f"环境探测失败：{info['error']}")
    else:
        info = json.loads(out[-1])
        if info["platform"] != "tpu" or info["n"] != args.tp:
            problems.append(
                f"期望 {args.tp} 个 TPU 设备，实际 {info['n']} 个 {info['platform']} 设备；"
                "如果之前的运行被打断过，先确认没有残留的服务进程占着 TPU"
            )
    for p in problems:
        run.log(f"  问题：{p}")
    run.record("F0", "FAIL" if problems else "PASS", problems=problems, devices=info)
    return not problems


def http(srv: rb.Server, path: str, payload: dict | None = None) -> str:
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(
        srv.base + path, data=data, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read().decode()


def capture_profile(run: rb.Run, srv: rb.Server, args) -> dict:
    """在预热好的服务上抓 prefill、decode 各几步的 profile。

    按阶段抓取时，服务端在阶段切换或者凑够步数时自己停；等它回到 idle，没回去就手动停。
    """
    out = run.out / f"profile_{srv.backend}"
    shutil.rmtree(out, ignore_errors=True)
    http(srv, "/start_profile", {**PROFILE_REQUEST, "output_dir": str(out.resolve())})
    rb.run_bench(run, srv, args, PROFILE_LOAD, tag="_profile")
    deadline = time.time() + PROFILE_WAIT_SECONDS
    while json.loads(http(srv, "/profile_status"))["status"] != "idle":
        if time.time() > deadline:
            http(srv, "/stop_profile")
            run.log(f"  profile 等了 {PROFILE_WAIT_SECONDS} 秒没有自己停，已手动停止")
            break
        time.sleep(10)
    files = [p for p in out.rglob("*") if p.is_file()]
    if sum(p.stat().st_size for p in files) > PROFILE_MAX_MB * 2**20:
        for p in files:
            if not p.name.endswith(".xplane.pb"):
                p.unlink()
        files = [p for p in files if p.exists()]
    if not any(p.name.endswith(".xplane.pb") for p in files):
        raise RuntimeError(f"{out} 里没有生成 xplane.pb")
    size_mb = round(sum(p.stat().st_size for p in files) / 2**20, 1)
    stages = sorted({p.relative_to(out).parts[0] for p in files})
    return {"dir": out.name, "stages": stages, "files": len(files), "size_mb": size_mb}


def measure_perf(run: rb.Run, srv: rb.Server, args) -> tuple[dict, list[str]]:
    """两种输入长度各压测一组，然后抓 profile。只记录，出错的项写进返回的错误列表。"""
    bench, errors = {}, []
    for cfg in BENCH_SWEEP:
        key = str(cfg["input_len"])
        try:
            bench[key] = rb.run_bench(run, srv, args, cfg, tag=f"_in{key}")
            run.log(f"  bench_serving（{srv.backend}，输入 {key}）：{bench[key]}")
        except Exception as e:  # noqa: BLE001 - 一组失败，另一组和 profile 照样做
            errors.append(f"bench_serving（输入 {key}）：{type(e).__name__}: {e}")
    try:
        profile = capture_profile(run, srv, args)
        run.log(f"  profile（{srv.backend}）：{profile}")
    except Exception as e:  # noqa: BLE001
        profile = {}
        errors.append(f"profile：{type(e).__name__}: {e}")
    for e in errors:
        run.log(f"  性能测量出错：{e}")
    return {"bench": bench, "profile": profile, "perf_errors": errors}, errors


def stage_f1(run: rb.Run, args) -> None:
    def f1(srv, ready, checks):
        run.log(f"  加载摘要（只记录）：{checks['load_summary']}")
        status, details = "PASS", {"load_summary": checks["load_summary"]}
        try:
            details["needles"] = rb.run_needles(run, srv, NEEDLES)
            missed = [
                n["target_tokens"] for n in details["needles"] if in_budget(n) and not n["found"]
            ]
            if missed:
                run.log(f"  预算内 fa 没找到：{missed}（问题在测试或模型本身，不算 QSA）")
                status = "WARN"
        except Exception as e:  # noqa: BLE001 - needle 出错时，性能测量照样做
            details["error"] = f"needle 出错：{type(e).__name__}: {e}"
            status = "FAIL"
        perf, errors = measure_perf(run, srv, args)
        if errors and status == "PASS":
            status = "WARN"
        run.record("F1", status, **details, **perf)

    rb.stage_launch(run, args, "fa", [("F1", f1)])


def judge(n: int, fa: dict | None, qsa: dict) -> tuple[str, str]:
    fa_found = None if fa is None else fa["found"]
    if qsa["found"]:
        return "PASS", "qsa_sparse 找到"
    if n < INDEXER_BUDGET:
        if fa_found:
            return "FAIL", "预算内两边数学上相同，fa 找到而 qsa_sparse 没找到"
        return "WARN", "预算内两边都没找到，问题在测试或模型本身"
    if fa_found:
        return "WARN", "超出预算，fa 找到而 qsa_sparse 没找到，选块可疑"
    return "WARN", "超出预算，两边都没找到，下不了结论"


def stage_f2(run: rb.Run, args) -> None:
    def f2(srv, ready, checks):
        status, details = "PASS", {}
        try:
            needles = rb.run_needles(run, srv, NEEDLES)
            fa = {n["target_tokens"]: n for n in run.results.get("F1", {}).get("needles", [])}
            verdicts = {}
            for qsa in needles:
                n = qsa["target_tokens"]
                verdicts[n] = judge(n, fa.get(n), qsa)
                run.log(f"  needle {n}：{verdicts[n][0]}，{verdicts[n][1]}")
            status = max((v[0] for v in verdicts.values()), key=RANK.get)
            details.update(needles=needles, verdicts=verdicts)
        except Exception as e:  # noqa: BLE001 - needle 出错时，性能测量照样做
            details["error"] = f"needle 出错：{type(e).__name__}: {e}"
            status = "FAIL"
        perf, errors = measure_perf(run, srv, args)
        if errors and status == "PASS":
            status = "WARN"
        run.record("F2", status, **details, **perf)

    rb.stage_launch(run, args, "qsa_sparse", [("F2", f2)])


def in_budget(needle: dict) -> bool:
    return needle["target_tokens"] < INDEXER_BUDGET


def write_report(run: rb.Run, repo: dict) -> Path:
    r = run.results
    lines = [
        "# Qwen3.8-Flash-Next 补充测试：超出预算后的稀疏选块（PLE 关闭）",
        "",
        f"代码：{repo.get('branch')} @ {repo.get('head')}",
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
    fa = {n["target_tokens"]: n for n in r.get("F1", {}).get("needles", [])}
    qsa = {n["target_tokens"]: n for n in r.get("F2", {}).get("needles", [])}
    verdicts = {int(k): v for k, v in r.get("F2", {}).get("verdicts", {}).items()}
    if fa or qsa:
        lines += [
            "",
            f"| needle | 实际 token 数 | 预算（{INDEXER_BUDGET}）| fa | qsa_sparse | 判定 |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
        for n, _ in NEEDLES:
            a, b = fa.get(n), qsa.get(n)
            tokens = (a or b or {}).get("prompt_tokens", "-")
            budget = "以内" if n < INDEXER_BUDGET else "超出"
            verdict = "，".join(verdicts[n]) if n in verdicts else "-"
            lines += [
                f"| {n} | {tokens} | {budget} | {rb.needle_status(a)} | {rb.needle_status(b)} | "
                f"{verdict} |"
            ]
    lines += perf_table(r)
    path = run.out / "report.md"
    path.write_text("\n".join(lines) + "\n")
    return path


def perf_table(r: dict) -> list[str]:
    rows = [
        (backend, key, m)
        for s, backend in (("F1", "fa"), ("F2", "qsa_sparse"))
        for key, m in r.get(s, {}).get("bench", {}).items()
    ]
    lines = []
    if rows:
        lines += [
            "",
            "性能（只记录；猜测 QSA 每步按最长上下文付费时，qsa_sparse 的 TPOT 基本不随输入长度变）",
            "",
            "| 后端 | 输入长度 | 完成数 | TTFT 中位数 ms | TPOT 中位数 ms | 输出吞吐 tok/s | 总吞吐 tok/s |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
        for backend, key, m in rows:
            cells = [m.get(k, "-") for k in ("completed", "median_ttft_ms", "median_tpot_ms")]
            cells += [m.get(k, "-") for k in ("output_throughput", "total_throughput")]
            cells = [f"{c:.1f}" if isinstance(c, float) else c for c in cells]
            lines += [f"| {backend} | {key} | " + " | ".join(map(str, cells)) + " |"]
    for s in ("F1", "F2"):
        c = r.get(s, {})
        if c.get("profile"):
            lines += [f"- {s} profile：{c['profile']}"]
        lines += [f"- {s} 性能测量出错：{e}" for e in c.get("perf_errors", [])]
    return lines


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--model-path", required=True, help="完整 checkpoint 所在的本地目录")
    ap.add_argument("--repo", default=os.getcwd(), help="sglang-jax 仓库根目录（默认当前目录）")
    ap.add_argument("--out", default="bringup-followup", help="输出目录")
    ap.add_argument("--port", type=int, default=30000, help="服务端口")
    ap.add_argument("--tp", type=int, default=rb.EXPECTED_DEVICES, help="张量并行度（默认 8）")
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
            print(shlex.join(rb.server_cmd(args, backend)), end="\n\n")
        return 0

    # SSH 断开时照 Ctrl-C 处理：先停掉服务，再写报告、打包
    signal.signal(signal.SIGHUP, signal.default_int_handler)
    run = rb.Run(Path(args.out))
    stages = {s.strip().upper() for s in args.stages.split(",")}
    # 重跑 fa 那次时，F2 的判定用的是旧的 fa 结果，一起清掉
    for s in stages | ({"F2"} if "F1" in stages else set()):
        run.results.pop(s, None)
    run.save()
    run.log(f"要跑的阶段：{sorted(stages)}；输出目录：{run.out.resolve()}")

    f0_failed = False
    try:
        if "F0" in stages and not stage_f0(run, args):
            f0_failed = True
            stages = set()
        if "F1" in stages:
            stage_f1(run, args)
        if "F2" in stages:
            stage_f2(run, args)
    finally:
        run.log(f"报告：{write_report(run, rb.git_info(args.repo))}")
        if f0_failed:
            run.log("环境检查（F0）没有通过：按上面列出的问题修好后重跑，这次不打包、不用发回")
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
