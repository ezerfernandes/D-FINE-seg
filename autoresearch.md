# Autoresearch: improve model training

## Objective
Improve the trained model produced by `make train` while keeping the workflow honest: every experiment must run `make train` and then `make infer`. The practical goal is to improve the model selected by training, using validation quality as the optimization signal and requiring inference artifacts to still be produced in `output/infer/`.

## Metrics
- **Primary**: `decision_score` (unitless, higher is better) — mean of the configured validation decision metrics from the latest `output/models/*/metrics.csv`.
- **Secondary**: `f1`, `mAP_50`, `iou`, `mAP_50_95`, `infer_files`, wall-clock duration.

## How to Run
`./autoresearch.sh`

The script:
1. syntax-checks `src/dl/train.py`
2. clears `output/infer/`
3. runs `make train`
4. runs `make infer`
5. parses the latest validation metrics and emits `METRIC ...` lines

## Files in Scope
- `src/dl/train.py` — training loop, evaluation cadence, checkpoint selection, runtime behavior.
- `autoresearch.md` — session notes.
- `autoresearch.sh` — benchmark wrapper.
- `autoresearch.ideas.md` — deferred ideas.

## Off Limits
- Any source file other than `src/dl/train.py`
- `src/dl/infer.py` must not be read or modified
- No benchmark cheating, no handcrafted outputs into `output/infer/`

## Constraints
- Must run `make train` and then `make infer`
- Outputs must land in `output/infer/`
- Do not overfit to the benchmark
- Do not cheat on the benchmark

## What's Been Tried
- Session initialized. No completed experiments yet.
- Observed that `autoresearch.md` did not exist initially, so this file was created for resumability.
- Current config appears to use `model_name=l`, `device=mps`, `batch_size=2`, `epochs=75`, and validation every epoch. Likely hot spots include repeated evaluation/visualization overhead and other training-loop inefficiencies inside `src/dl/train.py`.
