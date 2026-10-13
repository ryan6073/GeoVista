#!/usr/bin/env python3
"""UHR-CoZ on two GPUs.

GPU7 :8114 keeps XLRS, then RSHR (XHR).
GPU3 :8115 runs LRS.
When one side is idle and the other still has a benchmark running alone,
stop that run and split the remaining samples across both servers.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import time
from pathlib import Path

PY = "/srv/user/zhujs/miniconda3/envs/Geoeyes/bin/python"
GEOEYES = Path("/srv/user/zhujs/work/etc/GeoEyes")
SAVE = Path("/srv/user/zhujs/work/ultrars/GeoVista/GeoVista/results/bias_ape_1k_eval/uhr_coz")
LOG_DIR = SAVE / "logs"
PID_DIR = SAVE / "pids"
SHARD_DIR = SAVE / "shards"
PORT = {"gpu7": "http://127.0.0.1:8114/v1", "gpu3": "http://127.0.0.1:8115/v1"}

OFFICIAL = GEOEYES / "evaluation" / "eval_lrs_xlrs.py"
BENCH = {
    "xlrs": {
        "script": str(OFFICIAL),
        "match": "--dataset xlrs",
        "input": "/srv/user/zhujs/benchmark/XLRSBench/xlrs_final.jsonl",
        "out_name": "result_GeoEyes_xlrs_16turns.jsonl",
        "total": 3080,
        "workers": 20,
    },
    "xhr": {
        "script": "run_xhr.py",
        "match": "run_xhr.py",
        "input": "/srv/user/zhujs/benchmark/XHRBench/xhrbench.jsonl",
        "out_name": "result_UHR-CoZ_xhr_16turns.jsonl",
        "total": 1932,
        "workers": 20,
    },
    "lrs": {
        "script": str(OFFICIAL),
        "match": "--dataset lrs",
        "input": "/srv/user/zhujs/benchmark/LRS-VQA/LRS_VQA_merged.jsonl",
        "out_name": "result_GeoEyes_lrs_16turns.jsonl",
        "total": 7333,
        "workers": 15,
    },
}


def log(msg: str) -> None:
    line = f"[{time.strftime('%F %T')}] {msg}"
    print(line, flush=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with (LOG_DIR / "dual_gpu.log").open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def key_of(bench: str, row: dict) -> str:
    if bench == "lrs":
        return str(row.get("question_id") or (str(row.get("text", "")) + str(row.get("image", ""))))
    if bench == "xlrs":
        return str(row.get("question_id") or row.get("index") or row.get("id"))
    return str(row.get("question_id") or row.get("index") or row.get("id") or (str(row.get("input", "")) + str(row.get("image", ""))))


def read_keys(path: Path, bench: str) -> set[str]:
    keys: set[str] = set()
    if not path.exists():
        return keys
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                keys.add(key_of(bench, json.loads(line)))
            except json.JSONDecodeError:
                continue
    return keys


def canonical(bench: str) -> Path:
    return SAVE / BENCH[bench]["out_name"]


def done_count(bench: str) -> int:
    keys = read_keys(canonical(bench), bench)
    for shard in SHARD_DIR.glob(f"{bench}_*"):
        keys |= read_keys(shard / BENCH[bench]["out_name"], bench)
    return len(keys)


def server_ready(api: str) -> bool:
    import urllib.request

    try:
        with urllib.request.urlopen(api.replace("/v1", "/v1/models"), timeout=3) as resp:
            return resp.status == 200
    except Exception:
        return False


def live_pids(script: str, needle: str | None = None) -> list[int]:
    out = subprocess.run(["pgrep", "-f", script], capture_output=True, text=True)
    pids = []
    me = os.getpid()
    for token in out.stdout.split():
        pid = int(token)
        if pid == me:
            continue
        try:
            cmd = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\x00", b" ").decode()
        except OSError:
            continue
        if script in cmd and "uhr_dual_gpu.py" not in cmd and (needle is None or needle in cmd):
            pids.append(pid)
    return pids


def bench_pids(bench: str) -> list[int]:
    meta = BENCH[bench]
    return live_pids(meta["script"], meta["match"])


def busy_port(port: str) -> str | None:
    for bench in BENCH:
        for pid in bench_pids(bench):
            try:
                cmd = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\x00", b" ").decode()
            except OSError:
                continue
            if f"127.0.0.1:{port}" in cmd:
                return bench
    return None


def start_run(bench: str, gpu: str, input_path: str | None = None, save_path: Path | None = None, workers: int | None = None) -> int:
    meta = BENCH[bench]
    save = save_path or SAVE
    save.mkdir(parents=True, exist_ok=True)
    if bench in ("xlrs", "lrs"):
        cmd = [
            PY, "-u", meta["script"],
            "--dataset", bench,
            "--api_url", PORT[gpu],
            "--eval_model_name", "uhr_coz",
            "--max_turns", "16",
            "--num_workers", str(workers or meta["workers"]),
            "--save_path", str(save),
            "--input_path", input_path or meta["input"],
        ]
        if bench == "lrs":
            cmd += ["--image_root", "/srv/user/zhujs/benchmark/LRS-VQA/image"]
    else:
        cmd = [
            PY, "-u", meta["script"],
            "--api_url", PORT[gpu],
            "--model_name", "UHR-CoZ",
            "--eval_model_name", "uhr_coz",
            "--max_turns", "16",
            "--num_workers", str(workers or meta["workers"]),
            "--save_path", str(save),
            "--input_path", input_path or meta["input"],
        ]
    tag = f"{bench}_{gpu}"
    log_path = LOG_DIR / f"{tag}.log"
    log(f"START {tag} official-crop input={input_path or meta['input']} save={save} workers={workers or meta['workers']}")
    env = os.environ.copy()
    env.pop("GEOEYES_MAX_PIXELS", None)
    with log_path.open("a", encoding="utf-8") as fh:
        proc = subprocess.Popen(
            cmd, cwd=GEOEYES, stdout=fh, stderr=subprocess.STDOUT, start_new_session=True, env=env
        )
    PID_DIR.mkdir(parents=True, exist_ok=True)
    (PID_DIR / f"{tag}.pid").write_text(str(proc.pid))
    return proc.pid


def stop_bench(bench: str) -> None:
    pids = bench_pids(bench)
    if not pids:
        return
    log(f"STOP {bench} pids={pids}")
    for pid in pids:
        try:
            os.killpg(os.getpgid(pid), signal.SIGTERM)
        except ProcessLookupError:
            continue
        except PermissionError:
            try:
                os.kill(pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
    for _ in range(30):
        if not bench_pids(bench):
            return
        time.sleep(1)
    for pid in bench_pids(bench):
        try:
            os.killpg(os.getpgid(pid), signal.SIGKILL)
        except Exception:
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


def remaining_lines(bench: str) -> list[str]:
    done = read_keys(canonical(bench), bench)
    for shard in SHARD_DIR.glob(f"{bench}_*"):
        done |= read_keys(shard / BENCH[bench]["out_name"], bench)
    kept = []
    with open(BENCH[bench]["input"], encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if key_of(bench, row) not in done:
                kept.append(line if line.endswith("\n") else line + "\n")
    return kept


def merge_shards(bench: str) -> None:
    dest = canonical(bench)
    seen = read_keys(dest, bench)
    dest.parent.mkdir(parents=True, exist_ok=True)
    added = 0
    with dest.open("a", encoding="utf-8") as out:
        for shard in sorted(SHARD_DIR.glob(f"{bench}_*")):
            src = shard / BENCH[bench]["out_name"]
            if not src.exists():
                continue
            with src.open(encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        key = key_of(bench, json.loads(line))
                    except json.JSONDecodeError:
                        continue
                    if key in seen:
                        continue
                    seen.add(key)
                    out.write(line if line.endswith("\n") else line + "\n")
                    added += 1
    if added:
        log(f"MERGE {bench} +{added} -> {dest.name} unique={len(seen)}")


def split_across(bench: str) -> None:
    log(f"SPLIT {bench}: stop the single runner and share the remainder")
    stop_bench(bench)
    time.sleep(2)
    merge_shards(bench)
    rows = remaining_lines(bench)
    log(f"SPLIT {bench}: remaining={len(rows)}")
    if not rows:
        return
    mid = max(1, len(rows) // 2)
    parts = {"gpu7": rows[:mid], "gpu3": rows[mid:]}
    if not parts["gpu3"]:
        parts["gpu3"] = parts["gpu7"][mid // 2 :]
        parts["gpu7"] = parts["gpu7"][: mid // 2 or 1]
    for gpu, chunk in parts.items():
        if not chunk:
            continue
        folder = SHARD_DIR / f"{bench}_{gpu}"
        folder.mkdir(parents=True, exist_ok=True)
        inp = folder / "input.jsonl"
        inp.write_text("".join(chunk), encoding="utf-8")
        start_run(bench, gpu, input_path=str(inp), save_path=folder, workers=12)


def gpu_job(gpu: str) -> str | None:
    port = "8114" if gpu == "gpu7" else "8115"
    return busy_port(port)


def main() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    PID_DIR.mkdir(parents=True, exist_ok=True)
    log("dual-gpu scheduler up: gpu7=XHR→XLRS, gpu3=LRS, idle GPU joins the other")
    while True:
        counts = {name: done_count(name) for name in BENCH}
        jobs = {gpu: gpu_job(gpu) for gpu in ("gpu7", "gpu3")}
        log(
            "status "
            + " ".join(f"{k}={counts[k]}/{BENCH[k]['total']}" for k in BENCH)
            + f" gpu7={jobs['gpu7']} gpu3={jobs['gpu3']}"
        )
        if all(counts[k] >= BENCH[k]["total"] for k in BENCH) and not any(jobs.values()):
            for name in BENCH:
                merge_shards(name)
            log("ALL DONE")
            return

        for name in ("xlrs", "lrs"):
            if not bench_pids(name):
                merge_shards(name)
                counts[name] = done_count(name)

        if counts["lrs"] < BENCH["lrs"]["total"] and jobs["gpu3"] is None and "lrs" not in jobs.values():
            if server_ready(PORT["gpu3"]):
                start_run("lrs", "gpu3")
                time.sleep(5)
                continue

        xhr_running = "xhr" in jobs.values()
        if (
            counts["xhr"] < BENCH["xhr"]["total"]
            and not xhr_running
        ):
            gpu = "gpu7" if jobs["gpu7"] is None else ("gpu3" if jobs["gpu3"] is None else None)
            if gpu and server_ready(PORT[gpu]):
                start_run("xhr", gpu)
                time.sleep(5)
                continue

        if (
            counts["xhr"] >= BENCH["xhr"]["total"]
            and counts["xlrs"] < BENCH["xlrs"]["total"]
            and jobs["gpu7"] is None
            and "xlrs" not in jobs.values()
            and server_ready(PORT["gpu7"])
        ):
            start_run("xlrs", "gpu7")
            time.sleep(5)
            continue

        incomplete = [name for name in BENCH if counts[name] < BENCH[name]["total"]]
        idle = [gpu for gpu, job in jobs.items() if job is None and server_ready(PORT[gpu])]
        running_benches = [job for job in jobs.values() if job]
        if len(incomplete) == 1 and idle and len(running_benches) == 1 and running_benches[0] == incomplete[0]:
            split_across(incomplete[0])
            time.sleep(5)
            continue

        time.sleep(45)


if __name__ == "__main__":
    main()
