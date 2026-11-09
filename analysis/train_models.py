"""
TRN-1: static-dataset model training for Talking to Bridges.

Run from the project root, as a module:

    python -m analysis.train_models

Reads   data/processed/features_dataset.csv        (one row per recording)
Writes  models/damage_model_bundle.joblib           (models + feature lists + metadata)
        reports/trn1/training_report.md             (human-readable evaluation report)
        reports/trn1/metrics.json                   (all numbers, machine-readable)
        reports/trn1/*.csv, *.png                   (per-case results, out-of-fold predictions, plots)

Two targets
  Model A: Undamaged vs Damaged           (all recordings)
  Model B: severity 1 / 2 / 3 mm          (damaged recordings only)

Evaluation: leave-one-case-out (LOCO). All recordings of one case stay on the same
side of every split. Predictions from all folds are pooled and the metrics are
computed once on the pooled predictions (a single left-out case has one class only),
and per-case results are reported too.

Inputs are never label / file-name / case / sampling-rate columns.
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

LOG = logging.getLogger("ttb.train")

SEED = 42
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = PROJECT_ROOT / "data" / "processed" / "features_dataset.csv"
DEFAULT_MODEL_DIR = PROJECT_ROOT / "models"
DEFAULT_REPORT_DIR = PROJECT_ROOT / "reports" / "trn1"
BUNDLE_NAME = "damage_model_bundle.joblib"

# ---- column roles ---------------------------------------------------------
CASE_COL = "case_id"
SEVERITY_COL = "damage_level_mm"      # 1, 2, 3 ; empty / 0 for healthy
CONDITION_COL = "condition"           # "Damaged" / "Undamaged" (optional, used for a consistency check)

# A column is NEVER a model input if its name matches any of these (case-insensitive).
# This covers metadata, labels, case ids, file names and sampling rate / recording length
# (the EDA showed sampling rate alone reveals the class).
FORBIDDEN_PATTERNS = [
    r"(^|_)id$", r"^id(_|$)", r"recording", r"case", r"source", r"file", r"path", r"name",
    r"test_type", r"session", r"specimen", r"beam", r"location", r"^m\d$",
    r"condition", r"damage", r"label", r"target", r"severity", r"class",
    r"sampl", r"(^|_)fs(_|$)", r"duration", r"n_samples", r"n_points", r"record_length", r"^length", r"^time", r"timestamp",
]
FORBIDDEN_RE = re.compile("|".join(FORBIDDEN_PATTERNS), re.IGNORECASE)

FREQ_RE = re.compile(r"^f[1-9]_hz$", re.IGNORECASE)
RATIO_RE = re.compile(r"(_over_|ratio|_rel$|_rel_)", re.IGNORECASE)

MODEL_NAMES = ["logreg", "random_forest", "gradient_boosting"]
SIMPLICITY_RANK = {"logreg": 0, "random_forest": 1, "gradient_boosting": 2}   # tie-break: simpler wins
FEATURE_SET_RANK = {"freq": 0, "freq_ratio": 1, "freq_dev": 1, "freq_amp": 2}
FREQ3 = ["f1_hz", "f2_hz", "f3_hz"]
# Amplitude features (Sensor_k_rms, _peak, ...) depend on sensor gain / DC offset and can reveal the
# recording session instead of the damage. They are evaluated as a diagnostic but never selected
# for the saved model unless --allow-amplitude is given.
AMPLITUDE_SET = "freq_amp"


# ---------------------------------------------------------------------------
# data loading and checks
# ---------------------------------------------------------------------------
def load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python -m analysis.build_features_dataset` first."
        )
    df = pd.read_csv(path)
    for col in (CASE_COL, SEVERITY_COL):
        if col not in df.columns:
            raise KeyError(f"Required column '{col}' is missing. Columns found: {list(df.columns)}")

    df["_severity"] = pd.to_numeric(df[SEVERITY_COL], errors="coerce").fillna(0).astype(int)
    if not set(df["_severity"].unique()) <= {0, 1, 2, 3}:
        raise ValueError(f"Unexpected damage levels: {sorted(df['_severity'].unique())}")
    df["_damaged"] = (df["_severity"] > 0).astype(int)

    # consistency with the text label, if there is one
    if CONDITION_COL in df.columns:
        txt = df[CONDITION_COL].astype(str).str.strip().str.lower()
        txt_damaged = txt.isin({"damaged", "damage", "1", "true"}).astype(int)
        mismatch = int((txt_damaged != df["_damaged"]).sum())
        if mismatch:
            raise ValueError(
                f"{mismatch} rows where '{CONDITION_COL}' disagrees with '{SEVERITY_COL}'. "
                "Fix the labels before training."
            )

    # every case must carry exactly one label, otherwise the case split is meaningless
    per_case = df.groupby(CASE_COL)["_severity"].nunique()
    bad = per_case[per_case > 1]
    if len(bad):
        raise ValueError(f"Cases with more than one damage level: {list(bad.index)}")
    return df


def select_feature_sets(df: pd.DataFrame, explicit: list[str] | None, extra_exclude: list[str]) -> dict[str, list[str]]:
    """Return {feature_set_name: [columns]}. Prints and audits every column choice."""
    if explicit:
        missing = [c for c in explicit if c not in df.columns]
        if missing:
            raise KeyError(f"--features contains unknown columns: {missing}")
        cands = list(explicit)
    else:
        cands = []
        for c in df.columns:
            if c.startswith("_") or c in extra_exclude:
                continue
            if FORBIDDEN_RE.search(c) and not FREQ_RE.match(c):
                continue
            if not pd.api.types.is_numeric_dtype(df[c]):
                continue
            if df[c].notna().sum() == 0 or df[c].nunique(dropna=True) < 2:
                continue
            cands.append(c)

    # hard guard: a forbidden name must never reach the models
    leaked = [c for c in cands if FORBIDDEN_RE.search(c) and not FREQ_RE.match(c)]
    if leaked:
        raise ValueError(f"Forbidden columns selected as features: {leaked}")

    freq = [c for c in cands if FREQ_RE.match(c)]
    if len(freq) < 3:
        raise KeyError(f"Expected f1_hz, f2_hz, f3_hz among the features; found {freq}")
    ratio = [c for c in cands if RATIO_RE.search(c)]
    freq_ratio = freq + [c for c in ratio if c not in freq]
    sets = {"freq": freq, "freq_ratio": freq_ratio, AMPLITUDE_SET: list(cands)}
    return sets


# ---------------------------------------------------------------------------
# models
# ---------------------------------------------------------------------------
def make_model(name: str) -> Pipeline:
    imputer = ("imputer", SimpleImputer(strategy="median"))
    if name == "logreg":
        return Pipeline([imputer, ("scaler", StandardScaler()),
                         ("clf", LogisticRegression(class_weight="balanced", max_iter=5000, C=1.0))])
    if name == "random_forest":
        return Pipeline([imputer, ("clf", RandomForestClassifier(
            n_estimators=300, min_samples_leaf=2, class_weight="balanced_subsample",
            random_state=SEED, n_jobs=-1))])
    if name == "gradient_boosting":
        return Pipeline([imputer, ("clf", GradientBoostingClassifier(
            n_estimators=150, learning_rate=0.05, max_depth=2, subsample=0.8, random_state=SEED))])
    raise ValueError(name)


def fit_model(name: str, X: pd.DataFrame, y: np.ndarray) -> Pipeline:
    model = make_model(name)
    if name == "gradient_boosting":
        # GradientBoostingClassifier has no class_weight: use balanced sample weights instead
        model.fit(X, y, clf__sample_weight=compute_sample_weight("balanced", y))
    else:
        model.fit(X, y)
    return model


def feature_importances(model: Pipeline, features: list[str]) -> dict[str, float]:
    clf = model.named_steps["clf"]
    if hasattr(clf, "feature_importances_"):
        imp = np.asarray(clf.feature_importances_, dtype=float)
    else:  # logistic regression on standardised inputs: |coef| is comparable across features
        imp = np.abs(np.asarray(clf.coef_, dtype=float)).mean(axis=0)
    total = imp.sum()
    imp = imp / total if total > 0 else imp
    return {f: float(v) for f, v in sorted(zip(features, imp), key=lambda kv: -kv[1])}


# ---------------------------------------------------------------------------
# leave-one-case-out evaluation
# ---------------------------------------------------------------------------
def loco_predict(model_name: str, X: pd.DataFrame, y: np.ndarray, groups: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Pooled out-of-fold predictions. Returns (class_probabilities, classes)."""
    classes = np.unique(y)
    proba = np.zeros((len(y), len(classes)))
    for tr, te in LeaveOneGroupOut().split(X, y, groups):
        if len(np.unique(y[tr])) < len(classes):
            raise RuntimeError("A training fold is missing a class; cannot continue.")
        m = fit_model(model_name, X.iloc[tr], y[tr])
        p = m.predict_proba(X.iloc[te])
        # map the model's own class order onto the global class order
        for j, c in enumerate(m.named_steps["clf"].classes_):
            proba[te, np.where(classes == c)[0][0]] = p[:, j]
    return proba, classes


def dev_features(d: pd.DataFrame, ref: dict, cols=FREQ3) -> pd.DataFrame:
    """Two-sided relative deviation from the healthy reference: |f - healthy| / healthy.
    A frequency drop (what a crack should cause) and a rise both count as abnormal."""
    return pd.DataFrame({f"dev_{c}": (d[c] - ref[c]).abs() / ref[c] for c in cols}, index=d.index)


def loco_predict_dev(model_name: str, df: pd.DataFrame, y: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """LOCO for Model A on deviation features. The healthy reference is the median of the healthy
    recordings in the TRAINING fold only, so a held-out healthy beam never sets its own baseline."""
    proba = np.zeros((len(y), 2))
    for tr, te in LeaveOneGroupOut().split(df, y, groups):
        trd = df.iloc[tr]
        healthy = trd[trd["_damaged"] == 0]
        if len(healthy) == 0 or len(np.unique(y[tr])) < 2:
            raise RuntimeError("A training fold is missing a class; cannot continue.")
        ref = {c: float(healthy[c].median()) for c in FREQ3}
        m = fit_model(model_name, dev_features(trd, ref), y[tr])
        p = m.predict_proba(dev_features(df.iloc[te], ref))
        for j, c in enumerate(m.named_steps["clf"].classes_):
            proba[te, int(c)] = p[:, j]
    return proba


def metrics_binary(y_true, proba_damaged, threshold: float = 0.5) -> dict:
    y_pred = (proba_damaged >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    rec_h = cm[0, 0] / max(cm[0].sum(), 1)
    rec_d = cm[1, 1] / max(cm[1].sum(), 1)
    try:
        auc = float(roc_auc_score(y_true, proba_damaged))
    except ValueError:
        auc = float("nan")
    return {
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
        "roc_auc": auc,
        "accuracy": float((y_pred == y_true).mean()),
        "recall_undamaged": float(rec_h),
        "recall_damaged": float(rec_d),
        "confusion_matrix": cm.tolist(),          # rows = true [Undamaged, Damaged]
    }


def metrics_severity(y_true, y_pred) -> dict:
    labels = [1, 2, 3]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    err = np.abs(y_pred - y_true)
    return {
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", labels=labels)),
        "accuracy": float((y_pred == y_true).mean()),
        "mae_mm": float(err.mean()),
        "within_1mm": float((err <= 1).mean()),
        "confusion_matrix": cm.tolist(),          # rows = true [1, 2, 3] mm
    }


def evaluate_grid(target: str, df: pd.DataFrame, y: np.ndarray, groups: np.ndarray,
                  feature_sets: dict[str, list[str]]) -> tuple[list[dict], dict]:
    """Run every model x feature-set combination. Returns (rows, oof) where oof[(model,fs)] = (proba, classes)."""
    rows, oof = [], {}
    for fs_name, cols in feature_sets.items():
        X = df[cols]
        for mname in MODEL_NAMES:
            t0 = time.time()
            proba, classes = loco_predict(mname, X, y, groups)
            if target == "A":
                met = metrics_binary(y, proba[:, list(classes).index(1)])
            else:
                met = metrics_severity(y, classes[proba.argmax(axis=1)])
            met.update(model=mname, feature_set=fs_name, n_features=len(cols), seconds=round(time.time() - t0, 1))
            rows.append(met)
            oof[(mname, fs_name)] = (proba, classes)
            LOG.info("Model %s | %-17s | %-10s | bal.acc %.3f | f1 %.3f | %.1fs",
                     target, mname, fs_name, met["balanced_accuracy"], met["f1_macro"], met["seconds"])
    return rows, oof


def pick_best(rows: list[dict], primary: str) -> dict:
    # highest primary metric; ties (within 0.005) go to the simpler model, then the smaller feature set
    best = max(r[primary] for r in rows)
    tied = [r for r in rows if best - r[primary] <= 0.005]
    return sorted(tied, key=lambda r: (SIMPLICITY_RANK[r["model"]], FEATURE_SET_RANK[r["feature_set"]]))[0]


def per_case_table(df: pd.DataFrame, target: str, y, proba, classes, threshold: float = 0.5) -> pd.DataFrame:
    if target == "A":
        pred = (proba[:, list(classes).index(1)] >= threshold).astype(int)
    else:
        pred = classes[proba.argmax(axis=1)]
    out = pd.DataFrame({CASE_COL: df[CASE_COL].values, "true": y, "pred": pred})
    if target == "A":
        out["p_damaged"] = proba[:, list(classes).index(1)]
    rows = []
    for case, d in out.groupby(CASE_COL, sort=True):
        row = {
            CASE_COL: case,
            "true": int(d["true"].iloc[0]),
            "n_recordings": int(len(d)),
            "pct_correct": float((d["true"] == d["pred"]).mean() * 100),
            "majority_pred": int(d["pred"].value_counts().idxmax()),
        }
        if target == "A":
            row["mean_p_damaged"] = float(d["p_damaged"].mean())
        rows.append(row)
    tab = pd.DataFrame(rows)
    tab["case_correct"] = tab["true"] == tab["majority_pred"]
    return tab


# ---------------------------------------------------------------------------
# reporting helpers
# ---------------------------------------------------------------------------
def md_table(df: pd.DataFrame, floatfmt: str = "{:.3f}") -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, r in df.iterrows():
        cells = [floatfmt.format(v) if isinstance(v, (float, np.floating)) else str(v) for v in r.values]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def cm_md(cm, labels, title) -> str:
    d = pd.DataFrame(cm, index=[f"true {l}" for l in labels], columns=[f"pred {l}" for l in labels]).reset_index()
    d = d.rename(columns={"index": title})
    return md_table(d, "{:.0f}")


def save_confusion_png(cm, labels, title, path: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    cm = np.asarray(cm)
    fig, ax = plt.subplots(figsize=(4, 3.6))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)), labels)
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title, fontsize=9)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="TRN-1 model training (leave-one-case-out).")
    ap.add_argument("--data", type=Path, default=DEFAULT_DATA)
    ap.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    ap.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    ap.add_argument("--features", nargs="*", default=None,
                    help="Use exactly these columns as the 'freq_amp' feature set (overrides auto-detection).")
    ap.add_argument("--exclude", nargs="*", default=[], help="Extra columns to exclude from the features.")
    ap.add_argument("--threshold-a", type=float, default=0.5,
                    help="Decision threshold on P(damaged) for Model A (default 0.5). Raising it trades missed damage "
                         "for fewer false alarms on healthy beams. This is a design choice, not a tuned optimum.")
    ap.add_argument("--allow-amplitude", action="store_true",
                    help="Let the amplitude feature set (Sensor_* rms/peak/...) be selected for the saved model. "
                         "Off by default: amplitude features can encode the recording session.")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
    warnings.filterwarnings("ignore", category=UserWarning)
    np.random.seed(SEED)
    t_start = time.time()

    df = load_dataset(args.data)
    sets = select_feature_sets(df, args.features, args.exclude)
    groups_all = df[CASE_COL].astype(str).values

    n_cases = df[CASE_COL].nunique()
    healthy_cases = df.loc[df["_damaged"] == 0, CASE_COL].nunique()
    LOG.info("Loaded %d recordings, %d cases (%d healthy, %d damaged).", len(df), n_cases, healthy_cases, n_cases - healthy_cases)
    if healthy_cases < 2:
        raise RuntimeError("Need at least 2 healthy cases for leave-one-case-out.")
    for k, v in sets.items():
        LOG.info("Feature set %-10s (%2d): %s", k, len(v), ", ".join(v))
    LOG.info("Columns deliberately NOT used as inputs: %s",
             ", ".join(c for c in df.columns if c not in sets[AMPLITUDE_SET] and not c.startswith("_")))

    args.report_dir.mkdir(parents=True, exist_ok=True)
    args.model_dir.mkdir(parents=True, exist_ok=True)

    # ---------------- Model A: undamaged vs damaged ----------------
    yA = df["_damaged"].values
    rowsA, oofA = evaluate_grid("A", df, yA, groups_all, sets)
    for mname in MODEL_NAMES:   # Model A on two-sided deviation-from-healthy features
        t0 = time.time()
        pr_ = loco_predict_dev(mname, df, yA, groups_all)
        met = metrics_binary(yA, pr_[:, 1])
        met.update(model=mname, feature_set="freq_dev", n_features=len(FREQ3), seconds=round(time.time() - t0, 1))
        rowsA.append(met)
        oofA[(mname, "freq_dev")] = (pr_, np.array([0, 1]))
        LOG.info("Model A | %-17s | %-10s | bal.acc %.3f | f1 %.3f", mname, "freq_dev", met["balanced_accuracy"], met["f1_macro"])
    eligible = lambda rows: [r for r in rows if args.allow_amplitude or r["feature_set"] != AMPLITUDE_SET]
    bestA = dict(pick_best(eligible(rowsA), "balanced_accuracy"))   # copy: the grid table stays at threshold 0.5
    probaA, classesA = oofA[(bestA["model"], bestA["feature_set"])]
    if not 0.0 < args.threshold_a < 1.0:
        raise ValueError("--threshold-a must be between 0 and 1")
    pdA = probaA[:, list(classesA).index(1)]
    sweep = []
    for t in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
        m_ = metrics_binary(yA, pdA, t)
        sweep.append({"threshold": t, "recall_undamaged": m_["recall_undamaged"],
                      "recall_damaged": m_["recall_damaged"], "balanced_accuracy": m_["balanced_accuracy"]})
    if args.threshold_a != 0.5:   # report the selected model at the chosen operating point
        bestA.update(metrics_binary(yA, pdA, args.threshold_a))
    caseA = per_case_table(df, "A", yA, probaA, classesA, threshold=args.threshold_a)
    caseA.to_csv(args.report_dir / "model_a_per_case.csv", index=False)
    pd.DataFrame({CASE_COL: df[CASE_COL], "true_damaged": yA,
                  "p_damaged": probaA[:, list(classesA).index(1)]}).to_csv(args.report_dir / "model_a_oof_predictions.csv", index=False)

    # ---------------- Model B: severity (damaged only) ----------------
    dmg = df[df["_damaged"] == 1].reset_index(drop=True)
    yB = dmg["_severity"].values
    if set(np.unique(yB)) != {1, 2, 3}:
        raise RuntimeError(f"Model B needs levels 1, 2 and 3; found {sorted(set(yB))}")
    rowsB, oofB = evaluate_grid("B", dmg, yB, dmg[CASE_COL].astype(str).values, sets)
    bestB = pick_best(eligible(rowsB), "f1_macro")
    probaB, classesB = oofB[(bestB["model"], bestB["feature_set"])]
    caseB = per_case_table(dmg, "B", yB, probaB, classesB)
    caseB.to_csv(args.report_dir / "model_b_per_case.csv", index=False)
    pd.DataFrame({CASE_COL: dmg[CASE_COL], "true_mm": yB,
                  "pred_mm": classesB[probaB.argmax(axis=1)]}).to_csv(args.report_dir / "model_b_oof_predictions.csv", index=False)

    # ---------------- trivial baselines (so the numbers can be judged) ----------------
    majority_acc_A = float(max(yA.mean(), 1 - yA.mean()))
    majority_acc_B = float(pd.Series(yB).value_counts(normalize=True).max())

    # ---------------- final fit on ALL data + bundle ----------------
    colsB = sets[bestB["feature_set"]]
    a_transform = None
    if bestA["feature_set"] == "freq_dev":
        ref = {c: float(df.loc[df["_damaged"] == 0, c].median()) for c in FREQ3}
        a_transform = {"type": "abs_rel_dev", "reference": ref, "columns": FREQ3}
        colsA = FREQ3                                   # raw inputs the caller must supply
        XA_train = dev_features(df, ref)
        final_A = fit_model(bestA["model"], XA_train, yA)
        impA = feature_importances(final_A, list(XA_train.columns))
    else:
        colsA = sets[bestA["feature_set"]]
        final_A = fit_model(bestA["model"], df[colsA], yA)
        impA = feature_importances(final_A, colsA)
    final_B = fit_model(bestB["model"], dmg[colsB], yB)
    impB = feature_importances(final_B, colsB)

    medians = {c: float(df[c].median()) for c in sorted(set(colsA) | set(colsB))}
    bundle = {
        "version": 1,
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "sklearn_version": sklearn.__version__,
        "model_a": final_A, "model_a_name": bestA["model"], "features_a": colsA, "importances_a": impA, "a_transform": a_transform,
        "model_b": final_B, "model_b_name": bestB["model"], "features_b": colsB, "importances_b": impB,
        "classes_a": {0: "Undamaged", 1: "Damaged"},
        "threshold_a": float(args.threshold_a),
        "severity_levels_mm": [1, 2, 3],
        "feature_medians": medians,
        "forbidden_input_patterns": FORBIDDEN_PATTERNS,
        "training_summary": {
            "n_recordings": int(len(df)), "n_cases": int(n_cases), "n_healthy_cases": int(healthy_cases),
            "n_damaged_cases": int(n_cases - healthy_cases),
            "loco_model_a": {k: bestA[k] for k in ("balanced_accuracy", "f1_macro", "roc_auc", "recall_undamaged", "recall_damaged")},
            "loco_model_b": {k: bestB[k] for k in ("balanced_accuracy", "f1_macro", "mae_mm", "within_1mm")},
        },
    }
    joblib.dump(bundle, args.model_dir / BUNDLE_NAME)

    # ---------------- metrics.json + report ----------------
    def clean(rows):
        return [{k: v for k, v in r.items()} for r in rows]
    (args.report_dir / "metrics.json").write_text(json.dumps({
        "model_a_grid": clean(rowsA), "model_b_grid": clean(rowsB),
        "model_a_selected": {"model": bestA["model"], "feature_set": bestA["feature_set"]},
        "model_b_selected": {"model": bestB["model"], "feature_set": bestB["feature_set"]},
        "baseline_majority_accuracy": {"model_a": majority_acc_A, "model_b": majority_acc_B},
        "importances_a": impA, "importances_b": impB,
        "threshold_a": args.threshold_a, "threshold_sweep_model_a": sweep,
    }, indent=2))
    save_confusion_png(bestA["confusion_matrix"], ["Undamaged", "Damaged"], "Model A (pooled LOCO)", args.report_dir / "model_a_confusion.png")
    save_confusion_png(bestB["confusion_matrix"], ["1 mm", "2 mm", "3 mm"], "Model B (pooled LOCO)", args.report_dir / "model_b_confusion.png")

    def grid_md(rows, key):
        d = pd.DataFrame(rows)
        cols = ["model", "feature_set", "n_features", "balanced_accuracy", "f1_macro"] + (["roc_auc"] if key == "A" else ["mae_mm", "within_1mm"])
        return md_table(d[cols].sort_values("balanced_accuracy" if key == "A" else "f1_macro", ascending=False))

    caseA_show = caseA.copy()
    caseA_show["true"] = caseA_show["true"].map({0: "Undamaged", 1: "Damaged"})
    caseA_show["majority_pred"] = caseA_show["majority_pred"].map({0: "Undamaged", 1: "Damaged"})
    caseB_show = caseB.copy()
    def best_of(rows, fs, key):
        v = [r[key] for r in rows if r["feature_set"] == fs]
        return max(v) if v else float("nan")
    ampA, frqA = best_of(rowsA, AMPLITUDE_SET, "balanced_accuracy"), best_of(rowsA, "freq_ratio", "balanced_accuracy")
    ampB, frqB = best_of(rowsB, AMPLITUDE_SET, "f1_macro"), best_of(rowsB, "freq_ratio", "f1_macro")
    leak_note = (
        f"- Amplitude features were evaluated but not allowed into the saved model"
        f"{' (overridden by --allow-amplitude)' if args.allow_amplitude else ''}. "
        f"Best Model A balanced accuracy: {ampA:.3f} with amplitude features vs {frqA:.3f} with frequency features only. "
        f"Best Model B macro F1: {ampB:.3f} vs {frqB:.3f}. "
        "A large gain from amplitude features is a warning sign of session leakage (gain/offset), not proof of better damage detection."
    )
    topA = ", ".join(f"{k} ({v:.2f})" for k, v in list(impA.items())[:5])
    topB = ", ".join(f"{k} ({v:.2f})" for k, v in list(impB.items())[:5])

    report = f"""# TRN-1 training report

Generated {bundle['created']} | scikit-learn {sklearn.__version__} | seed {SEED}

## Data
- {len(df)} recordings, {n_cases} cases: {healthy_cases} healthy, {n_cases - healthy_cases} damaged.
- Every evaluation below is leave-one-case-out: all recordings of a case are held out together.
- Metrics are computed once on the pooled out-of-fold predictions; per-case results follow.
- Inputs never include label, file-name, case or sampling-rate columns.

## Model A: Undamaged vs Damaged
Selected (frequency-based sets only unless --allow-amplitude): **{bestA['model']}** on feature set **{bestA['feature_set']}** ({bestA['n_features']} features).
`freq_dev` = two-sided deviation of f1-f3 from the healthy reference (median of the healthy training recordings), so a frequency drop and a rise both count as abnormal.

{grid_md(rowsA, 'A')}

Pooled result for the selected model: balanced accuracy {bestA['balanced_accuracy']:.3f}, macro F1 {bestA['f1_macro']:.3f}, ROC-AUC {bestA['roc_auc']:.3f}, recall on undamaged {bestA['recall_undamaged']:.3f}, recall on damaged {bestA['recall_damaged']:.3f}.
Always answering "Damaged" would score {majority_acc_A:.3f} plain accuracy but 0.500 balanced accuracy.

{cm_md(bestA['confusion_matrix'], ['Undamaged', 'Damaged'], 'Model A')}

Per case (majority vote of the case's recordings):

{md_table(caseA_show, '{:.2f}')}

Decision threshold on P(damaged): **{args.threshold_a}** (design choice). Trade-off for the selected model:

{md_table(pd.DataFrame(sweep))}

Top features: {topA}

**Caveat.** Only {healthy_cases} healthy cases exist. Each healthy fold is trained with a single other healthy beam, so the undamaged-class numbers rest on {healthy_cases} test cases and are very uncertain. Healthy and damaged recordings also come from differently named sets, so part of any "damaged vs healthy" separation may be setup or mounting differences, not damage.

## Model B: severity (1, 2, 3 mm), damaged recordings only
Selected (frequency-based sets only unless --allow-amplitude): **{bestB['model']}** on feature set **{bestB['feature_set']}** ({bestB['n_features']} features).

{grid_md(rowsB, 'B')}

Pooled result for the selected model: balanced accuracy {bestB['balanced_accuracy']:.3f}, macro F1 {bestB['f1_macro']:.3f}, mean absolute error {bestB['mae_mm']:.3f} mm, within 1 mm {bestB['within_1mm']:.3f}.
Always answering the most common level would score {majority_acc_B:.3f}.

{cm_md(bestB['confusion_matrix'], ['1 mm', '2 mm', '3 mm'], 'Model B')}

Per case (majority vote of the case's recordings):

{md_table(caseB_show, '{:.2f}')}

Top features: {topB}

**Caveat.** Severity depends on crack location as well as depth. Cracks near the free end barely change the first mode, so 1 mm cracks there can look healthy or like other levels. Expect weaker results there.

## Honest limits
{leak_note}
- The model was chosen among {len(rowsA)} combinations (Model A) and {len(rowsB)} (Model B) using the same leave-one-case-out scores that are reported, so the selected scores are slightly optimistic.
- Results apply to this 0.7 m cantilever beam, a single crack and a single boundary condition only.
- Do not present these figures as guarantees.
"""
    (args.report_dir / "training_report.md").write_text(report)

    LOG.info("=" * 70)
    LOG.info("Model A selected: %s / %s | balanced acc %.3f | recall healthy %.3f | recall damaged %.3f",
             bestA["model"], bestA["feature_set"], bestA["balanced_accuracy"], bestA["recall_undamaged"], bestA["recall_damaged"])
    LOG.info("Model B selected: %s / %s | balanced acc %.3f | macro F1 %.3f | MAE %.3f mm",
             bestB["model"], bestB["feature_set"], bestB["balanced_accuracy"], bestB["f1_macro"], bestB["mae_mm"])
    LOG.info("Saved bundle:  %s", args.model_dir / BUNDLE_NAME)
    LOG.info("Saved report:  %s", args.report_dir / "training_report.md")
    LOG.info("Total time %.1fs", time.time() - t_start)
    return 0


if __name__ == "__main__":
    sys.exit(main())