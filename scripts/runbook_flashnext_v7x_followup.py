"""Qwen3.8-Flash-Next 跑通测试的补充测试：超出 indexer 预算后，稀疏选块能不能把开头的内容找回来。

和 runbook_flashnext_v7x.py 用同一套启动参数（N-gram / PLE 层关闭），只测那份 runbook 没覆盖到
的部分。结束时留下一份报告和一个压缩包。

各阶段
  F0  环境检查：8 个 TPU 设备、--repo 是 sglang-jax 仓库根目录、代码包含 REQUIRED_COMMIT。
      不满足就停。
  F1  稠密注意力（fa）启动：
      - needle 四档：长文开头埋一个 6 位口令，末尾问它，约 500 / 1500 / 6000 / 14000 token。
        前两档在 indexer 预算（2048 token）以内，后两档超出。贪心生成 512 个 token（模型先在
        <think> 里思考再回答），记下完整输出、有没有找到口令、口令是否出现在 </think> 之后。
      - bench_serving，参数和 runbook 的 C4 相同。
  F2  稀疏注意力（qsa_sparse）启动：同样四档 needle。

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
  F1、F2 不管什么结果都不用自己排查，把压缩包原样发回。

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
import signal
import subprocess
import sys
import tarfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import runbook_flashnext_v7x as rb  # noqa: E402

STAGES = ("F0", "F1", "F2")
STAGE_INFO = {
    "F0": "环境检查：TPU 设备、仓库和代码版本",
    "F1": "fa：四档 needle（预算内两档必须找到）+ 压测",
    "F2": "qsa_sparse：四档 needle，超出预算后选块能不能找回口令",
}
rb.STAGE_INFO.update(STAGE_INFO)  # stage_launch 打印阶段标题时查这张表

INDEXER_BUDGET = 2048
NEEDLES = [(500, "615208"), *rb.NEEDLES]  # 500 档确认关掉 PLE 后模型本身能找回口令

RANK = {"PASS": 0, "WARN": 1, "FAIL": 2}


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


def stage_f1(run: rb.Run, args) -> None:
    def f1(srv, ready, checks):
        run.log(f"  加载摘要（只记录）：{checks['load_summary']}")
        needles = rb.run_needles(run, srv, NEEDLES)
        try:
            bench = rb.run_bench(run, srv, args)
        except Exception as e:  # noqa: BLE001 - 压测出错只记下来，不影响 needle 的结论
            bench = {"error": f"{type(e).__name__}: {e}"}
        run.log(f"  bench_serving（fa）：{bench}")
        missed = [n["target_tokens"] for n in needles if in_budget(n) and not n["found"]]
        if missed:
            run.log(f"  预算内 fa 没找到：{missed}（问题在测试或模型本身，不算 QSA）")
        status = "WARN" if missed or "error" in bench else "PASS"
        run.record("F1", status, needles=needles, bench=bench, load_summary=checks["load_summary"])

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
        needles = rb.run_needles(run, srv, NEEDLES)
        fa = {n["target_tokens"]: n for n in run.results.get("F1", {}).get("needles", [])}
        verdicts = {}
        for qsa in needles:
            n = qsa["target_tokens"]
            verdicts[n] = judge(n, fa.get(n), qsa)
            run.log(f"  needle {n}：{verdicts[n][0]}，{verdicts[n][1]}")
        status = max((v[0] for v in verdicts.values()), key=RANK.get)
        run.record("F2", status, needles=needles, verdicts=verdicts)

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
    lines += rb.bench_table(r.get("F1", {}).get("bench"), None)
    path = run.out / "report.md"
    path.write_text("\n".join(lines) + "\n")
    return path


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
