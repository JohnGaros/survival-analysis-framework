# Survival Modeling Framework (v1)

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python train.py --csv data.csv
```

- Artifacts in `artifacts/`:
  - `ph_flags.csv` — Schoenfeld tests per covariate
  - `<model>/..._surv.npy` & `_risk.npy` — per-fold predictions
  - `model_metrics.csv`, `model_summary.csv` — cross-validated results
- Versioned fitted pipelines in `models/` as `.joblib`.

## Notes

- Categorical strata are used in the lifelines Stratified Cox wrapper.
- Event-balanced CV (StratifiedKFold on event indicator).
- Metrics: Harrell's C (IPCW), IBS, time-dependent AUC.
- Extend with DeepSurv later by implementing `DeepSurvWrapper`.
