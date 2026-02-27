#!/usr/bin/env python3
import argparse
import csv
import datetime as dt
import os
import re
import subprocess
import sys
import tempfile


def parse_wall_time(raw: str) -> float:
    raw = raw.strip()
    if not raw:
        return 0.0
    parts = raw.split(":")
    try:
        if len(parts) == 1:
            return float(parts[0])
        if len(parts) == 2:
            minutes = float(parts[0])
            seconds = float(parts[1])
            return minutes * 60.0 + seconds
        if len(parts) == 3:
            hours = float(parts[0])
            minutes = float(parts[1])
            seconds = float(parts[2])
            return hours * 3600.0 + minutes * 60.0 + seconds
    except ValueError:
        return 0.0
    return 0.0


def parse_time_file(path: str):
    wall = 0.0
    rss_kb = -1
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if "Elapsed (wall clock) time" in line:
                match = re.search(r"Elapsed \\(wall clock\\) time.*: ([0-9:.]+)", line)
                if match:
                    wall = parse_wall_time(match.group(1))
            elif "Maximum resident set size" in line:
                match = re.search(r"Maximum resident set size.*: ([0-9]+)", line)
                if match:
                    rss_kb = int(match.group(1))
    return wall, rss_kb


def find_checksum(output: str) -> str:
    for line in output.splitlines():
        if line.startswith("CHECKSUM="):
            return line.split("=", 1)[1].strip()
    return ""


def get_git_sha(repo_root: str) -> str:
    try:
        result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo_root, capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except Exception:
        return ""


def main() -> int:
    parser = argparse.ArgumentParser(description="Run wo-coulomb CPU benchmarks with fixed pinning and output CSV.")
    parser.add_argument("--binary", default=None, help="Path to wo-coulomb-bench.x")
    parser.add_argument("--sizes", default="small,medium", help="Comma-separated sizes to run (small,medium)")
    parser.add_argument("--threads", type=int, default=1, help="OpenMP threads per run")
    parser.add_argument("--iters", type=int, default=1, help="Iterations inside the benchmark binary")
    parser.add_argument("--output", default="results.csv", help="Output CSV filename (relative to bench dir)")
    parser.add_argument("--no-pin", action="store_true", help="Do not enforce OpenMP pinning env vars")
    args = parser.parse_args()

    bench_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(bench_dir, ".."))
    binary = args.binary or os.path.join(repo_root, "build", "wo-coulomb-bench.x")

    if not os.path.exists(binary):
        print(f"Benchmark binary not found: {binary}")
        print("Build with: ./configure_and_build.sh (from wo-coulomb/) or cmake --build build/")
        return 1

    sizes = [s.strip() for s in args.sizes.split(",") if s.strip()]
    if not sizes:
        print("No benchmark sizes specified.")
        return 1

    output_path = os.path.join(bench_dir, args.output)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    use_time = os.path.exists("/usr/bin/time")
    if not use_time:
        print("/usr/bin/time not found; memory usage will be reported as -1.")

    env = os.environ.copy()
    env.setdefault("OMP_NUM_THREADS", str(args.threads))
    env.setdefault("WO_DETERMINISTIC", "1")
    if not args.no_pin:
        env["OMP_PROC_BIND"] = env.get("OMP_PROC_BIND", "close")
        env["OMP_PLACES"] = env.get("OMP_PLACES", "cores")
        env["OMP_DYNAMIC"] = env.get("OMP_DYNAMIC", "FALSE")
        env["OMP_SCHEDULE"] = env.get("OMP_SCHEDULE", "static")

    git_sha = get_git_sha(repo_root)
    write_header = not os.path.exists(output_path)

    with open(output_path, "a", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        if write_header:
            writer.writerow([
                "timestamp",
                "size",
                "threads",
                "iters",
                "wall_seconds",
                "max_rss_kb",
                "checksum",
                "git_sha",
            ])

        for size in sizes:
            cmd = [binary, "--size", size, "--threads", str(args.threads), "--iters", str(args.iters)]
            if not use_time:
                result = subprocess.run(cmd, capture_output=True, text=True, env=env)
                wall_seconds = 0.0
                max_rss_kb = -1
            else:
                with tempfile.NamedTemporaryFile(delete=False) as tmp:
                    time_file = tmp.name
                time_cmd = ["/usr/bin/time", "-v", "-o", time_file, "--"] + cmd
                result = subprocess.run(time_cmd, capture_output=True, text=True, env=env)
                wall_seconds, max_rss_kb = parse_time_file(time_file)
                os.unlink(time_file)

            if result.returncode != 0:
                print("Benchmark failed:")
                print(result.stdout)
                print(result.stderr)
                return result.returncode

            checksum = find_checksum(result.stdout)
            timestamp = dt.datetime.utcnow().isoformat(timespec="seconds") + "Z"
            writer.writerow([
                timestamp,
                size,
                args.threads,
                args.iters,
                f"{wall_seconds:.6f}",
                max_rss_kb,
                checksum,
                git_sha,
            ])
            csvfile.flush()
            print(f"{size}: wall={wall_seconds:.3f}s rss_kb={max_rss_kb} checksum={checksum}")

    print(f"Results written to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
