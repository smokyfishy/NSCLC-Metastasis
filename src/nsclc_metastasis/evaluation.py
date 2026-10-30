"""Evaluation protocol, metrics and plots.

Protocol (see README, "Evaluation protocol"):

1. One stratified hold-out split by patient (``split.test_size``, seed
   ``split.seed``). Test patients are not touched until step 4.
2. ``split.outer_cv_folds``-fold stratified CV on the training partition. In
   each outer fold the *entire* stacked model — including its internal OOF
   cross-fitting, SMOTE, imputation, LASSO/PCA and autoencoder — is fitted on
   the outer-training rows and scored on the outer-validation rows.
3. The final model is fitted on the full training partition.
4. The final model is scored once on the held-out test patients.

All split memberships are written to ``split_manifest.csv``.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split

from .config import Config
from .data import MultiOmicsDataset
from .stacking import MultiOmicsStackingModel, safe_n_splits

log = logging.getLogger(__name__)

METRIC_NAMES = [
    "accuracy",
    "balanced_accuracy",
    "roc_auc",
    "pr_auc",
    "precision",
    "recall",
    "f1",
    "cohen_kappa",
]


def compute_metrics(y_true, y_score, threshold: float = 0.5) -> dict:
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    y_pred = (y_score >= threshold).astype(int)
    two_classes = len(np.unique(y_true)) == 2
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "n": int(len(y_true)),
        "n_positive": int(y_true.sum()),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)) if two_classes else float("nan"),
        "roc_auc": float(roc_auc_score(y_true, y_score)) if two_classes else float("nan"),
        "pr_auc": float(average_precision_score(y_true, y_score)) if two_classes else float("nan"),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "cohen_kappa": float(cohen_kappa_score(y_true, y_pred)) if two_classes else float("nan"),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "threshold": threshold,
    }


def holdout_split(ids: pd.Index, y: pd.Series, test_size: float, seed: int):
    train_ids, test_ids = train_test_split(
        np.asarray(ids), test_size=test_size, stratify=y.loc[ids].to_numpy(), random_state=seed
    )
    return pd.Index(train_ids, name=ids.name), pd.Index(test_ids, name=ids.name)


@dataclass
class ExperimentResult:
    model: MultiOmicsStackingModel
    split_manifest: pd.DataFrame
    cv_fold_metrics: pd.DataFrame
    cv_summary: dict
    test_metrics: dict
    test_predictions: pd.DataFrame
    data_summary: dict


def _score_model(model: MultiOmicsStackingModel, X, y, threshold) -> tuple[dict, pd.DataFrame]:
    detail = model.predict_detail(X)
    scores = {"ensemble": compute_metrics(y, detail["ensemble_probability"], threshold)}
    prefix = "base_" + model.config.stacking.meta_features + "_"
    for m in model.modalities_:
        scores[m] = compute_metrics(y, detail[prefix + m], threshold)
    detail.insert(0, "true_label", y.loc[detail.index].to_numpy())
    return scores, detail


def run_experiment(dataset: MultiOmicsDataset, cfg: Config, skip_cv: bool = False) -> ExperimentResult:
    if dataset.y is None:
        raise ValueError("Training requires a labelled dataset.")
    X, y = dataset.X, dataset.y
    thr = cfg.stacking.threshold
    train_ids, test_ids = holdout_split(X.index, y, cfg.split.test_size, cfg.split.seed)
    manifest = pd.DataFrame({"patient_id": X.index, "label": y.to_numpy()})
    manifest["partition"] = np.where(manifest["patient_id"].isin(test_ids), "test", "train")
    manifest["outer_cv_fold"] = -1

    X_tr, y_tr = X.loc[train_ids], y.loc[train_ids]
    X_te, y_te = X.loc[test_ids], y.loc[test_ids]
    assert not set(train_ids) & set(test_ids), "train/test overlap"

    fold_rows = []
    if not skip_cv:
        n_splits = safe_n_splits(y_tr, cfg.split.outer_cv_folds)
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=cfg.split.seed)
        for k, (tr, va) in enumerate(skf.split(X_tr, y_tr)):
            t0 = time.time()
            val_ids = X_tr.index[va]
            manifest.loc[manifest["patient_id"].isin(val_ids), "outer_cv_fold"] = k
            fold_model = MultiOmicsStackingModel(cfg).fit(X_tr.iloc[tr], y_tr.iloc[tr])
            scores, _ = _score_model(fold_model, X_tr.iloc[va], y_tr.iloc[va], thr)
            for model_name, s in scores.items():
                row = {"fold": k, "model": model_name}
                row.update({m: s[m] for m in METRIC_NAMES})
                row.update({f"cm_{k2}": v for k2, v in s["confusion_matrix"].items()})
                row["n_val"] = s["n"]
                row["n_val_positive"] = s["n_positive"]
                fold_rows.append(row)
            log.info("outer CV fold %d/%d finished in %.1fs", k + 1, n_splits, time.time() - t0)
    cv_df = pd.DataFrame(fold_rows)
    cv_summary = {}
    if not cv_df.empty:
        g = cv_df.groupby("model")[METRIC_NAMES]
        cv_summary = {
            name: {m: {"mean": float(g.mean().loc[name, m]), "std": float(g.std(ddof=1).loc[name, m])} for m in METRIC_NAMES}
            for name in g.mean().index
        }

    final = MultiOmicsStackingModel(cfg).fit(X_tr, y_tr)
    test_metrics, test_pred = _score_model(final, X_te, y_te, thr)

    return ExperimentResult(
        model=final,
        split_manifest=manifest,
        cv_fold_metrics=cv_df,
        cv_summary=cv_summary,
        test_metrics=test_metrics,
        test_predictions=test_pred,
        data_summary=dataset.summary(cfg.data.expected),
    )


# ---------------------------------------------------------------- reporting
def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    raise TypeError(type(o))


def write_json(obj, path: Path) -> None:
    path.write_text(json.dumps(obj, indent=2, default=_json_default), encoding="utf-8")


def save_plots(result: ExperimentResult, out_dir: Path) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_curve, roc_curve

    out_dir.mkdir(parents=True, exist_ok=True)
    pred = result.test_predictions
    y = pred["true_label"].to_numpy()
    model = result.model
    prefix = "base_" + model.config.stacking.meta_features + "_"
    series = {"Stacked ensemble": pred["ensemble_probability"]}
    labels = {"rs": "RNA-seq (DAE+RF)", "cn": "CNV RF", "mu": "Mutation RF", "pp": "RPPA RF"}
    for m in model.modalities_:
        series[labels.get(m, m)] = pred[prefix + m]
    paths = []
    if len(np.unique(y)) == 2:
        fig, ax = plt.subplots(figsize=(6.5, 6))
        for name, s in series.items():
            fpr, tpr, _ = roc_curve(y, s)
            ax.plot(fpr, tpr, lw=2 if name == "Stacked ensemble" else 1.2,
                    label=f"{name} (AUC {roc_auc_score(y, s):.2f})")
        ax.plot([0, 1], [0, 1], "--", color="grey", lw=1)
        ax.set(xlabel="False positive rate", ylabel="True positive rate",
               title=f"ROC — held-out test (n={len(y)}, positives={int(y.sum())})")
        ax.legend(loc="lower right", fontsize=8)
        p = out_dir / "roc_test.png"; fig.tight_layout(); fig.savefig(p, dpi=150); plt.close(fig); paths.append(p)

        fig, ax = plt.subplots(figsize=(6.5, 6))
        for name, s in series.items():
            prec, rec, _ = precision_recall_curve(y, s)
            ax.step(rec, prec, where="post", lw=2 if name == "Stacked ensemble" else 1.2,
                    label=f"{name} (AP {average_precision_score(y, s):.2f})")
        ax.axhline(y.mean(), ls="--", color="grey", lw=1, label=f"Prevalence {y.mean():.2f}")
        ax.set(xlabel="Recall", ylabel="Precision", title="Precision–recall — held-out test")
        ax.legend(loc="upper right", fontsize=8)
        p = out_dir / "pr_test.png"; fig.tight_layout(); fig.savefig(p, dpi=150); plt.close(fig); paths.append(p)

    cm = result.test_metrics["ensemble"]["confusion_matrix"]
    mat = np.array([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]])
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.imshow(mat, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(mat[i, j]), ha="center", va="center",
                    color="white" if mat[i, j] > mat.max() / 2 else "black", fontsize=14)
    ax.set_xticks([0, 1], ["Pred 0", "Pred 1"]); ax.set_yticks([0, 1], ["True 0", "True 1"])
    ax.set_title("Stacked ensemble — test confusion matrix")
    p = out_dir / "confusion_matrix_test.png"; fig.tight_layout(); fig.savefig(p, dpi=150); plt.close(fig); paths.append(p)

    imp = model.meta_feature_importance_
    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    ax.barh([labels.get(m, m) for m in imp.index[::-1]], imp.values[::-1], color="#4C72B0")
    ax.set(xlabel="Gini importance", title="Meta-classifier: modality importance")
    p = out_dir / "meta_feature_importance.png"; fig.tight_layout(); fig.savefig(p, dpi=150); plt.close(fig); paths.append(p)
    return paths


def write_reports(result: ExperimentResult, cfg: Config, out_dir: Path, extra: dict | None = None) -> None:
    from .modality_models import feature_importance

    out_dir.mkdir(parents=True, exist_ok=True)
    result.split_manifest.to_csv(out_dir / "split_manifest.csv", index=False)
    result.test_predictions.to_csv(out_dir / "test_predictions.csv", index_label="patient_id")
    result.model.oof_meta_features_.join(result.model.stacking_folds_).to_csv(
        out_dir / "train_oof_meta_features.csv", index_label="patient_id"
    )
    if not result.cv_fold_metrics.empty:
        result.cv_fold_metrics.to_csv(out_dir / "cv_fold_metrics.csv", index=False)
    for m, pipe in result.model.base_models_.items():
        feature_importance(pipe, result.model.feature_columns_[m]).head(50).to_csv(
            out_dir / f"feature_importance_{m}.csv", index=False
        )
    metrics = {
        "config_name": cfg.name,
        "data": result.data_summary,
        "split": {
            "n_train": int((result.split_manifest["partition"] == "train").sum()),
            "n_test": int((result.split_manifest["partition"] == "test").sum()),
            "n_train_positive": int(result.split_manifest.query("partition == 'train'")["label"].sum()),
            "n_test_positive": int(result.split_manifest.query("partition == 'test'")["label"].sum()),
            "seed": cfg.split.seed,
        },
        "cv_summary": result.cv_summary,
        "test_metrics": result.test_metrics,
        "meta_feature_importance": result.model.meta_feature_importance_.to_dict(),
    }
    if extra:
        metrics.update(extra)
    write_json(metrics, out_dir / "metrics.json")
    save_plots(result, out_dir)
