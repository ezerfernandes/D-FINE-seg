#!/bin/bash
set -euo pipefail

python -m py_compile src/dl/train.py
rm -rf output/infer
mkdir -p output/infer

make train
make infer

python - <<'PY'
from pathlib import Path
import numpy as np
import pandas as pd
import yaml

models_dir = Path('output/models')
model_dirs = sorted([p for p in models_dir.iterdir() if p.is_dir()], key=lambda p: p.stat().st_mtime)
if not model_dirs:
    raise SystemExit('No model directory found under output/models')
run_dir = model_dirs[-1]
metrics_path = run_dir / 'metrics.csv'
if not metrics_path.exists():
    raise SystemExit(f'Missing metrics file: {metrics_path}')

df = pd.read_csv(metrics_path, index_col=0)
if 'val' not in df.index:
    raise SystemExit(f"No 'val' row found in {metrics_path}")
val = df.loc['val']

with open('config.yaml', 'r') as f:
    cfg = yaml.safe_load(f)
metric_names = [m for m in cfg['train']['decision_metrics'] if m in val.index]
if not metric_names:
    raise SystemExit('No configured decision metrics found in metrics.csv')

decision_score = float(np.mean([float(val[m]) for m in metric_names]))
infer_files = sum(1 for p in Path('output/infer').rglob('*') if p.is_file())

print(f'METRIC decision_score={decision_score}')
for name in ['f1', 'mAP_50', 'iou', 'mAP_50_95']:
    if name in val.index:
        print(f'METRIC {name}={float(val[name])}')
print(f'METRIC infer_files={infer_files}')
PY
