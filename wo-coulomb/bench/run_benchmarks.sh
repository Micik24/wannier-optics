#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
./run_benchmarks.py "$@"
