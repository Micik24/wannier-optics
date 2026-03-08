#!/usr/bin/env python3
import argparse
import os
import shutil
import subprocess
from pathlib import Path


def parse_mapping(mapping_path: Path):
    entries = []
    with mapping_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "-->" not in line:
                continue
            left, right = line.split("-->")
            idx = int(left.strip())
            path = right.strip()
            entries.append((idx, path))
    return entries


def verify_mapping_files(mapping_path: Path, base_dir: Path):
    missing = []
    for idx, rel in parse_mapping(mapping_path):
        p = base_dir / rel
        if not p.exists():
            missing.append((idx, rel))
    return missing


def ensure_symlink(target: Path, link: Path):
    if link.exists() or link.is_symlink():
        return
    link.symlink_to(target)


def run_command(cmd, cwd: Path, env: dict, log_path: Path, time_path: Path | None = None):
    if time_path:
        full_cmd = ["/usr/bin/time", "-v", "-o", str(time_path), "--"] + cmd
    else:
        full_cmd = cmd
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.run(full_cmd, cwd=str(cwd), env=env, stdout=log, stderr=subprocess.STDOUT)
    return proc.returncode


def parse_coulomb_file(path: Path):
    data = {}
    header = None
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if header is None:
                header = line
                continue
            if line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 14:
                continue
            key = tuple(int(x) for x in parts[:13])
            val = float(parts[13])
            data[key] = val
    return header, data


def parse_transition_file(path: Path):
    data = {}
    count = None
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if count is None:
                count = int(line)
                continue
            if line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 8:
                continue
            key = (int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3]), int(parts[4]))
            vals = (float(parts[5]), float(parts[6]), float(parts[7]))
            data[key] = vals
    return count, data


def compare_scalar_maps(ref, new, abs_tol, rel_tol):
    missing = 0
    extra = 0
    max_abs = 0.0
    max_rel = 0.0
    for key, ref_val in ref.items():
        if key not in new:
            missing += 1
            continue
        val = new[key]
        diff = abs(val - ref_val)
        rel = diff / abs(ref_val) if ref_val != 0 else diff
        max_abs = max(max_abs, diff)
        max_rel = max(max_rel, rel)
    for key in new.keys():
        if key not in ref:
            extra += 1
    ok = (missing == 0 and extra == 0 and max_abs <= abs_tol and max_rel <= rel_tol)
    return ok, {"missing": missing, "extra": extra, "max_abs": max_abs, "max_rel": max_rel}


def compare_vector_maps(ref, new, abs_tol, rel_tol):
    missing = 0
    extra = 0
    max_abs = 0.0
    max_rel = 0.0
    for key, ref_vals in ref.items():
        if key not in new:
            missing += 1
            continue
        vals = new[key]
        for rv, v in zip(ref_vals, vals):
            diff = abs(v - rv)
            rel = diff / abs(rv) if rv != 0 else diff
            max_abs = max(max_abs, diff)
            max_rel = max(max_rel, rel)
    for key in new.keys():
        if key not in ref:
            extra += 1
    ok = (missing == 0 and extra == 0 and max_abs <= abs_tol and max_rel <= rel_tol)
    return ok, {"missing": missing, "extra": extra, "max_abs": max_abs, "max_rel": max_rel}


def verify_reference_set(reference_dir: Path, tutorial_dir: Path, w90_dir: Path):
    missing = []
    for name in ["input.ini", "vmapping.txt", "cmapping.txt", "COULOMB", "LOCALFIELDEFFECTS", "TRANSITION"]:
        if not (reference_dir / name).exists():
            missing.append(str(reference_dir / name))
    if not w90_dir.exists():
        missing.append(str(w90_dir))
    missing_v = []
    missing_c = []
    if (reference_dir / "vmapping.txt").exists():
        missing_v = verify_mapping_files(reference_dir / "vmapping.txt", tutorial_dir)
    if (reference_dir / "cmapping.txt").exists():
        missing_c = verify_mapping_files(reference_dir / "cmapping.txt", tutorial_dir)
    return missing, missing_v, missing_c


def run_tutorial_01_si_case(
    repo_root: Path,
    np: int,
    omp_threads: int,
    case_name: str,
    reference_dir: Path,
    mpirun_args: list[str],
    binary_path: Path,
    run_label: str,
):
    tutorial_dir = repo_root / "tutorial" / "01_Si"
    w90_dir = tutorial_dir / "w90_calculations"
    if not reference_dir.exists():
        return False, f"Missing reference directory: {reference_dir}"

    missing, missing_v, missing_c = verify_reference_set(reference_dir, tutorial_dir, w90_dir)
    if missing or missing_v or missing_c:
        msg = []
        if missing:
            msg.append(f"Missing required files/dirs: {missing[:3]}{'...' if len(missing)>3 else ''}")
        if missing_v:
            msg.append(f"Missing valence files: {missing_v[:3]}{'...' if len(missing_v)>3 else ''}")
        if missing_c:
            msg.append(f"Missing conduction files: {missing_c[:3]}{'...' if len(missing_c)>3 else ''}")
        return False, "\n".join(msg)

    case_dir_name = case_name if not run_label else f"{case_name}_{run_label}"
    run_root = repo_root / "wo-coulomb" / "bench" / "tutorial_runs" / f"01_Si_{case_dir_name}_np{np}_omp{omp_threads}"
    run_root.mkdir(parents=True, exist_ok=True)

    for name in ["input.ini", "vmapping.txt", "cmapping.txt"]:
        shutil.copy(reference_dir / name, run_root / name)

    ensure_symlink(w90_dir, run_root / "w90_calculations")

    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(omp_threads)
    env["WO_DETERMINISTIC"] = "1"
    env["WO_TIMING"] = "1"
    env["LD_LIBRARY_PATH"] = "/usr/lib/x86_64-linux-gnu"

    log_path = run_root / "run.log"
    time_path = run_root / "time.txt"

    cmd = ["mpirun", "-np", str(np)] + mpirun_args + [str(binary_path), "input.ini"]
    rc = run_command(cmd, run_root, env, log_path, time_path)
    if rc != 0:
        return False, f"wo-coulomb run failed with code {rc}. See {log_path}."

    # Compare outputs
    abs_tol = 1e-6
    rel_tol = 1e-5

    results = []
    # COULOMB
    ref_header, ref_coulomb = parse_coulomb_file(reference_dir / "COULOMB")
    new_header, new_coulomb = parse_coulomb_file(run_root / "COULOMB")
    ok, stats = compare_scalar_maps(ref_coulomb, new_coulomb, abs_tol, rel_tol)
    results.append(("COULOMB", ok, stats))

    # LOCALFIELDEFFECTS
    ref_header, ref_lfe = parse_coulomb_file(reference_dir / "LOCALFIELDEFFECTS")
    new_header, new_lfe = parse_coulomb_file(run_root / "LOCALFIELDEFFECTS")
    ok, stats = compare_scalar_maps(ref_lfe, new_lfe, abs_tol, rel_tol)
    results.append(("LOCALFIELDEFFECTS", ok, stats))

    # TRANSITION
    ref_count, ref_trans = parse_transition_file(reference_dir / "TRANSITION")
    new_count, new_trans = parse_transition_file(run_root / "TRANSITION")
    ok, stats = compare_vector_maps(ref_trans, new_trans, abs_tol, rel_tol)
    stats["ref_count"] = ref_count
    stats["new_count"] = new_count
    results.append(("TRANSITION", ok, stats))

    summary_lines = []
    all_ok = True
    for name, ok, stats in results:
        all_ok = all_ok and ok
        summary_lines.append(f"{name}: ok={ok} stats={stats}")

    return all_ok, "\n".join(summary_lines)


def run_tutorial_01_si(
    repo_root: Path,
    np: int,
    omp_threads: int,
    variant: str,
    mpirun_args: list[str],
    binary_path: Path,
    run_label: str,
):
    tutorial_dir = repo_root / "tutorial" / "01_Si"
    reference_dirs = []
    if variant in ("solution", "both"):
        reference_dirs.append(("solution", tutorial_dir / "solution"))
    if variant in ("converged", "both"):
        reference_dirs.append(("converged", tutorial_dir / "converged_calculation"))

    all_ok = True
    summaries = []
    for case_name, ref_dir in reference_dirs:
        ok, msg = run_tutorial_01_si_case(
            repo_root,
            np,
            omp_threads,
            case_name,
            ref_dir,
            mpirun_args,
            binary_path,
            run_label,
        )
        summaries.append(f"{case_name}:\n{msg}")
        all_ok = all_ok and ok
    return all_ok, "\n\n".join(summaries)


def main():
    parser = argparse.ArgumentParser(description="Run and verify wo-coulomb tutorial benchmarks.")
    parser.add_argument("--np", type=int, default=4, help="MPI processes")
    parser.add_argument("--omp", type=int, default=1, help="OpenMP threads")
    parser.add_argument("--tutorial", default="all", help="all or 01_Si")
    parser.add_argument("--variant", default="both", help="solution, converged, or both")
    parser.add_argument("--mpirun-args", default="", help="extra mpirun args, e.g. '--oversubscribe'")
    parser.add_argument(
        "--binary",
        default="",
        help="Path to wo-coulomb executable. Default: wo-coulomb/build/wo-coulomb.x",
    )
    parser.add_argument(
        "--run-label",
        default="",
        help="Optional suffix for output directory names (e.g. gpu).",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    binary_path = (
        Path(args.binary).resolve()
        if args.binary
        else (repo_root / "wo-coulomb" / "build" / "wo-coulomb.x")
    )

    if not binary_path.exists():
        print(f"Error: binary does not exist: {binary_path}")
        return 2

    if args.tutorial in ("all", "01_Si"):
        mpirun_args = args.mpirun_args.split() if args.mpirun_args else []
        ok, msg = run_tutorial_01_si(
            repo_root,
            args.np,
            args.omp,
            args.variant,
            mpirun_args,
            binary_path,
            args.run_label,
        )
        print("01_Si:")
        print(msg)
        if not ok:
            return 1

    if args.tutorial in ("all", "02_Wannier-Mott"):
        print("02_Wannier-Mott:")
        print("No wo-coulomb inputs in this tutorial (wo-optics only). Skipping.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
