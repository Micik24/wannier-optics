#!/bin/bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export WO_DETERMINISTIC="${WO_DETERMINISTIC:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
"$ROOT/build/tests/runTests.x" --gtest_filter=CorrectnessSmall.*:CorrectnessMedium.*
