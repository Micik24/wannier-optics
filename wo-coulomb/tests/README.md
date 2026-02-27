# wo-coulomb Test Policy

## Determinism
Tests default to deterministic OpenMP scheduling when `WO_DETERMINISTIC=1` (enabled in the test runner by default).

## Golden Updates
The small/medium correctness tests use golden outputs. To refresh them:

```bash
WO_UPDATE_GOLDEN=1 LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu ./tests/run_correctness.sh
```

## Tolerance Policy
For CPU correctness comparisons we use a mixed absolute + relative tolerance:

- Small test: `abs=1e-6`, `rel=5e-3`
- Medium test: `abs=1e-6`, `rel=2e-3`

The check is:

```
|value - reference| <= abs + rel * |reference|
```

This allows minor floating‑point differences while still catching regressions.
