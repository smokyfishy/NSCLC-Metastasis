"""Command-line interface.

    python -m nsclc_metastasis.cli validate-data --data data/raw/NSCLCMultiOmicsDataProcessed_reordered.csv
    python -m nsclc_metastasis.cli train    --data <csv> --config configs/baseline.yaml --model-dir artifacts/model
    python -m nsclc_metastasis.cli evaluate --model artifacts/model --data <csv>
    python -m nsclc_metastasis.cli predict  --model artifacts/model --data <csv> --out predictions.csv
    python -m nsclc_metastasis.cli legacy   --data <csv> --config configs/legacy_hyperparameters.yaml
    python -m nsclc_metastasis.cli make-synthetic --out data/processed/SYNTHETIC_demo.csv

The same commands are available as the ``nsclc-metastasis`` console script.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import pandas as pd

from .config import load_config
from .data import load_dataset

DEFAULT_CONFIG = "configs/baseline.yaml"
log = logging.getLogger("nsclc_metastasis")


def _cmd_validate(args) -> int:
    cfg = load_config(args.config)
    ds = load_dataset(args.data, cfg.data, require_label=not args.unlabeled)
    print(json.dumps(ds.summary(cfg.data.expected), indent=2))
    return 0


def _cmd_train(args) -> int:
    from .evaluation import run_experiment, write_reports
    from .inference import DISCLAIMER, save_model

    cfg = load_config(args.config)
    ds = load_dataset(args.data, cfg.data)
    synthetic = ds.X.index.astype(str).str.startswith("SYNTH-").all()
    reports = Path(args.reports_dir or f"reports/{cfg.name}{'_SYNTHETIC' if synthetic else ''}")
    t0 = time.time()
    result = run_experiment(ds, cfg, skip_cv=args.skip_cv)
    write_reports(result, cfg, reports, extra={
        "synthetic_data": bool(synthetic),
        "runtime_seconds": round(time.time() - t0, 1),
        "disclaimer": DISCLAIMER,
    })
    save_model(result.model, args.model_dir, ds, result.split_manifest)
    print(f"Model artifact: {args.model_dir}")
    print(f"Reports:        {reports}")
    if synthetic:
        print("NOTE: input data is SYNTHETIC; metrics are smoke-test values only.")
    _print_summary(result.cv_summary, result.test_metrics)
    return 0


def _print_summary(cv_summary: dict, test_metrics: dict) -> None:
    keys = ["roc_auc", "pr_auc", "balanced_accuracy", "f1", "cohen_kappa", "accuracy"]
    rows = []
    for name, s in test_metrics.items():
        row = {"model": name}
        for k in keys:
            row[f"test_{k}"] = round(s[k], 3)
            if name in cv_summary:
                row[f"cv_{k}"] = f"{cv_summary[name][k]['mean']:.3f}+/-{cv_summary[name][k]['std']:.3f}"
        rows.append(row)
    with pd.option_context("display.width", 200, "display.max_columns", 30):
        print(pd.DataFrame(rows).set_index("model"))


def _cmd_evaluate(args) -> int:
    from .evaluation import compute_metrics, write_json
    from .inference import load_model, load_split_manifest, predict_dataset

    model = load_model(args.model)
    ds = load_dataset(args.data, model.config.data)
    split = load_split_manifest(args.model)
    X, y = ds.X, ds.y
    scope = "all rows in file"
    if args.subset == "test":
        if split is None:
            print("ERROR: model has no split_manifest.csv; use --subset all.", file=sys.stderr)
            return 2
        test_ids = split.loc[split["partition"] == "test", "patient_id"].astype(str)
        missing = set(test_ids) - set(X.index)
        if missing:
            print(f"ERROR: {len(missing)} held-out test IDs are not in this file.", file=sys.stderr)
            return 2
        X, y = X.loc[test_ids], y.loc[test_ids]
        scope = "held-out test partition from training run"
    else:
        if split is not None and set(split["patient_id"].astype(str)) & set(X.index):
            log.warning("Evaluating on rows that include training patients; metrics will be optimistic.")
    ds.X, ds.y = X, y
    pred = predict_dataset(model, ds)
    prefix = "base_" + model.config.stacking.meta_features + "_"
    metrics = {"scope": scope, "ensemble": compute_metrics(y, pred["ensemble_probability"], model.config.stacking.threshold)}
    for m in model.modalities_:
        metrics[m] = compute_metrics(y, pred[prefix + m], model.config.stacking.threshold)
    out = Path(args.reports_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_json(metrics, out / f"evaluate_{args.subset}.json")
    pred.insert(0, "true_label", y.to_numpy())
    pred.to_csv(out / f"evaluate_{args.subset}_predictions.csv")
    print(json.dumps({k: (v if k == "scope" else {kk: v[kk] for kk in ("n", "roc_auc", "pr_auc", "balanced_accuracy", "f1")})
                      for k, v in metrics.items()}, indent=2))
    return 0


def _cmd_predict(args) -> int:
    from .inference import DISCLAIMER, load_model, predict_dataset

    model = load_model(args.model)
    ds = load_dataset(args.data, model.config.data, require_label=False)
    pred = predict_dataset(model, ds)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pred.to_csv(args.out)
    print(f"Wrote {len(pred)} predictions to {args.out}")
    print(DISCLAIMER)
    return 0


def _cmd_legacy(args) -> int:
    from .legacy import LEGACY_WARNING, run_legacy

    cfg = load_config(args.config)
    ds = load_dataset(args.data, cfg.data)
    df = run_legacy(ds, cfg)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(LEGACY_WARNING)
    print(df[["modality", "reducer", "accuracy", "roc_auc", "cohen_kappa"]].to_string(index=False))
    return 0


def _cmd_synthetic(args) -> int:
    from .synthetic import make_synthetic_dataset

    n, n_features = args.n, None
    if args.reported_shape:
        # Same dimensions as the cohort described in the write-up (220 x 1,558; 33 positive).
        n, n_features = 220, {"rs": 401, "cn": 813, "mu": 126, "pp": 218}
        args.positive_fraction = 33 / 220
    df = make_synthetic_dataset(n_samples=n, positive_fraction=args.positive_fraction, seed=args.seed,
                                missing_fraction=args.missing_fraction, n_features=n_features)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"Wrote SYNTHETIC dataset ({len(df)} rows) to {args.out}. Not real patient data.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="nsclc-metastasis", description=__doc__.split("\n\n")[0])
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("validate-data", help="Check schema and report counts vs. the write-up")
    s.add_argument("--data", required=True)
    s.add_argument("--config", default=DEFAULT_CONFIG)
    s.add_argument("--unlabeled", action="store_true", help="Do not require the label column")
    s.set_defaults(func=_cmd_validate)

    s = sub.add_parser("train", help="Hold-out split, nested CV, final fit, reports and artifact")
    s.add_argument("--data", required=True)
    s.add_argument("--config", default=DEFAULT_CONFIG)
    s.add_argument("--model-dir", default="artifacts/model")
    s.add_argument("--reports-dir", default=None, help="Default: reports/<config name>")
    s.add_argument("--skip-cv", action="store_true", help="Skip outer CV (faster; test-set metrics only)")
    s.set_defaults(func=_cmd_train)

    s = sub.add_parser("evaluate", help="Score a saved model on labelled data")
    s.add_argument("--model", required=True, help="Model artifact directory")
    s.add_argument("--data", required=True)
    s.add_argument("--subset", choices=["test", "all"], default="test",
                   help="'test' = held-out patients recorded at training time (default)")
    s.add_argument("--reports-dir", default="reports/evaluate")
    s.set_defaults(func=_cmd_evaluate)

    s = sub.add_parser("predict", help="Predict for (possibly unlabelled) patients")
    s.add_argument("--model", required=True)
    s.add_argument("--data", required=True)
    s.add_argument("--out", default="reports/predictions.csv")
    s.set_defaults(func=_cmd_predict)

    s = sub.add_parser("legacy", help="Re-run the ORIGINAL leaky per-modality protocol (comparison only)")
    s.add_argument("--data", required=True)
    s.add_argument("--config", default="configs/legacy_hyperparameters.yaml")
    s.add_argument("--out", default="reports/legacy/legacy_leaky_metrics.csv")
    s.set_defaults(func=_cmd_legacy)

    s = sub.add_parser("make-synthetic", help="Write a SYNTHETIC dataset for smoke tests")
    s.add_argument("--out", default="data/processed/SYNTHETIC_demo.csv")
    s.add_argument("--n", type=int, default=150)
    s.add_argument("--positive-fraction", type=float, default=0.15)
    s.add_argument("--missing-fraction", type=float, default=0.0)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--reported-shape", action="store_true",
                   help="Use the write-up's dimensions (220 patients, 401/813/126/218 features)")
    s.set_defaults(func=_cmd_synthetic)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
