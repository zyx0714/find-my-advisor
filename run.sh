#!/bin/bash
set -e
cd "$(dirname "$0")"
exec /Users/zhouyaxin/miniforge3/envs/ml/bin/uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
