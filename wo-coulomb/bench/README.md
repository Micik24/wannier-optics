# wo-coulomb CPU Benchmarks

This folder provides a small synthetic benchmark to establish CPU baselines before GPU work.

## Build
From `wo-coulomb/`:

```bash
./configure_and_build.sh
```

The benchmark binary is `build/wo-coulomb-bench.x`.

## Run
Default small + medium runs (deterministic, pinned):

```bash
./bench/run_benchmarks.sh --threads 1 --iters 1
```

Results are written to `bench/results.csv` with wall time and max RSS.

## Notes
- `WO_DETERMINISTIC=1` enforces static OpenMP scheduling.
- Pinning is applied via `OMP_PROC_BIND`, `OMP_PLACES`, `OMP_DYNAMIC`, and `OMP_SCHEDULE`.
- Use `--no-pin` if you want to manage affinity externally (e.g., with `mpirun` or `taskset`).
