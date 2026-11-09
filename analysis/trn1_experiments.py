"""
TRN-1 improvement experiments. Run AFTER train_models.py, from the project root:

    python -m analysis.trn1_experiments

Everything is leave-one-case-out on frequency-based features only (no amplitude / sampling-rate / label
columns). Writes reports/trn1/experiments.md and prints it. Nothing here changes the saved model.
NOTE: comparing many variants on the same folds is optimistic; use the results to see what helps,
then report the chosen setup's score with that caveat.
"""
from __future__ import annotations
import argparse, logging, sys, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from analysis.train_models import CASE_COL, DEFAULT_DATA, DEFAULT_REPORT_DIR, SEED, load_dataset, md_table

LOG = logging.getLogger("ttb.exp")


def add_freq_features(df):
    d = df.copy()
    d["f3_over_f2"] = d["f3_hz"] / d["f2_hz"]
    d["f1_x_f2_ratio"] = d["f2_over_f1"] * d["f3_over_f1"]
    d["log_f2_over_f1"] = np.log(d["f2_over_f1"])
    d["log_f3_over_f1"] = np.log(d["f3_over_f1"])
    return d


def clf(name):
    imp = ("imp", SimpleImputer(strategy="median"))
    if name == "logreg":
        return Pipeline([imp, ("sc", StandardScaler()), ("m", LogisticRegression(class_weight="balanced", max_iter=5000))])
    if name == "svm_rbf":
        return Pipeline([imp, ("sc", StandardScaler()), ("m", SVC(kernel="rbf", C=3.0, gamma="scale", class_weight="balanced", probability=True, random_state=SEED))])
    if name == "knn15":
        return Pipeline([imp, ("sc", StandardScaler()), ("m", KNeighborsClassifier(n_neighbors=15, weights="distance"))])
    if name == "extra_trees":
        return Pipeline([imp, ("m", ExtraTreesClassifier(n_estimators=300, min_samples_leaf=2, class_weight="balanced_subsample", random_state=SEED, n_jobs=-1))])
    if name == "random_forest":
        return Pipeline([imp, ("m", RandomForestClassifier(n_estimators=300, min_samples_leaf=2, class_weight="balanced_subsample", random_state=SEED, n_jobs=-1))])
    raise ValueError(name)


def reg(name):
    imp = ("imp", SimpleImputer(strategy="median"))
    if name == "ridge":
        return Pipeline([imp, ("sc", StandardScaler()), ("m", Ridge(alpha=1.0))])
    if name == "rf_reg":
        return Pipeline([imp, ("m", RandomForestRegressor(n_estimators=300, min_samples_leaf=2, random_state=SEED, n_jobs=-1))])
    if name == "gb_reg":
        return Pipeline([imp, ("m", GradientBoostingRegressor(n_estimators=150, learning_rate=0.05, max_depth=2, subsample=0.8, random_state=SEED))])
    raise ValueError(name)


def loco(make, X, y, groups, proba=True):
    out = np.zeros((len(y), 2 if proba else 1)) if proba is not False else None
    pred = np.zeros(len(y)) if not proba else None
    classes = np.unique(y)
    P = np.zeros((len(y), len(classes)))
    for tr, te in LeaveOneGroupOut().split(X, y, groups):
        m = make()
        m.fit(X.iloc[tr], y[tr])
        if proba:
            p = m.predict_proba(X.iloc[te])
            for j, c in enumerate(m.classes_):
                P[te, list(classes).index(c)] = p[:, j]
        else:
            pred[te] = m.predict(X.iloc[te])
    return (P, classes) if proba else pred



def loco_dev(make, df, cols=("f1_hz", "f2_hz", "f3_hz")):
    """Two-sided deviation-from-healthy features. The healthy reference is the median of the healthy
    recordings in the TRAINING fold only (so a held-out healthy beam never sets its own baseline)."""
    y, g = df["_damaged"].values, df[CASE_COL].astype(str).values
    P = np.zeros(len(y))
    for tr, te in LeaveOneGroupOut().split(df, y, g):
        trd = df.iloc[tr]
        h = trd[trd["_damaged"] == 0][list(cols)].median()
        tf = lambda d: pd.DataFrame({f"dev_{c}": (d[c] - h[c]).abs() / h[c] for c in cols}, index=d.index)
        m = make()
        m.fit(tf(trd), y[tr])
        P[te] = m.predict_proba(tf(df.iloc[te]))[:, list(m.classes_).index(1)]
    return P

def case_vote_acc(cases, y, pred):
    d = pd.DataFrame({"c": cases, "y": y, "p": pred})
    r = d.groupby("c").apply(lambda g: int(g["y"].iloc[0] == g["p"].value_counts().idxmax()))
    return int(r.sum()), int(len(r))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=DEFAULT_DATA)
    ap.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    warnings.filterwarnings("ignore")
    df = add_freq_features(load_dataset(args.data))
    t0 = time.time()
    base = ["f1_hz", "f2_hz", "f3_hz", "f2_over_f1", "f3_over_f1"]
    ext = base + ["f3_over_f2", "log_f2_over_f1", "log_f3_over_f1"]
    sets = {"base(5)": base, "extended(8)": ext, "no_f2": [c for c in ext if c not in ("f2_hz", "f2_over_f1", "f3_over_f2", "log_f2_over_f1")]}
    md = ["# TRN-1 experiments (leave-one-case-out, frequency features only)\n"]

    # --- 0. diagnostic: frequencies by case and sampling rate ---
    g = df.groupby(CASE_COL).agg(sampling_rate_hz=("sampling_rate_hz", "median"), f1=("f1_hz", "mean"), f2=("f2_hz", "mean"),
                                 f3=("f3_hz", "mean"), f2_std=("f2_hz", "std"), n=("f1_hz", "size")).reset_index()
    md += ["## 0. Mean frequencies per case (check the sampling-rate question)\n", md_table(g, "{:.3f}"), ""]

    # --- A: undamaged vs damaged ---
    yA, grpA = df["_damaged"].values, df[CASE_COL].astype(str).values
    rowsA = []
    for sname, cols in sets.items():
        for m in ["logreg", "svm_rbf", "knn15", "extra_trees", "random_forest"]:
            P, cl = loco(lambda m=m: clf(m), df[cols], yA, grpA)
            pd_ = P[:, 1]
            for thr in (0.5,):
                pr = (pd_ >= thr).astype(int)
                cm = confusion_matrix(yA, pr, labels=[0, 1])
                rowsA.append(dict(features=sname, model=m, bal_acc=balanced_accuracy_score(yA, pr), auc=roc_auc_score(yA, pd_),
                                  recall_healthy=cm[0, 0] / cm[0].sum(), recall_damaged=cm[1, 1] / cm[1].sum()))
            LOG.info("A %-12s %-14s done", sname, m)
    A = pd.DataFrame(rowsA).sort_values("bal_acc", ascending=False)
    md += ["## 1. Model A: undamaged vs damaged (threshold 0.5)\n", md_table(A), "",
           "`no_f2` removes every f2-based feature. If it scores close to the others, the result does not depend on f2 alone.\n"]

    # threshold trade-off for the best non-trivial model on the extended set
    bestA = A[A.features == "extended(8)"].iloc[0]["model"]
    P, cl = loco(lambda: clf(bestA), df[ext], yA, grpA)
    thr_rows = []
    for thr in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8):
        pr = (P[:, 1] >= thr).astype(int)
        cm = confusion_matrix(yA, pr, labels=[0, 1])
        thr_rows.append(dict(threshold=thr, recall_healthy=cm[0, 0] / cm[0].sum(), recall_damaged=cm[1, 1] / cm[1].sum(), bal_acc=balanced_accuracy_score(yA, pr)))
    md += [f"### Threshold trade-off ({bestA}, extended features)\n", md_table(pd.DataFrame(thr_rows)), ""]

    # two-sided deviation features (a frequency shift in EITHER direction counts as abnormal)
    dev_rows = []
    for m in ["logreg", "random_forest", "extra_trees"]:
        pdv = loco_dev(lambda m=m: clf(m), df)
        for thr in (0.5, 0.7):
            pr = (pdv >= thr).astype(int)
            cm = confusion_matrix(yA, pr, labels=[0, 1])
            dev_rows.append(dict(model=m, threshold=thr, bal_acc=balanced_accuracy_score(yA, pr), auc=roc_auc_score(yA, pdv),
                                 recall_healthy=cm[0, 0] / cm[0].sum(), recall_damaged=cm[1, 1] / cm[1].sum(),
                                 **{f"{c} called damaged": float(pr[(grpA == c)].mean()) for c in ("3mm_M3", "2mm_M6")}))
        LOG.info("A deviation features %-14s done", m)
    md += ["### Model A with two-sided deviation-from-healthy features\n",
           "Features are |f - healthy median| / healthy median for f1-f3, with the healthy median taken from the training folds only. "
           "A drop in frequency (what a crack physically does) now counts as abnormal, as well as a rise. "
           "Last two columns: share of that case's recordings called damaged.\n", md_table(pd.DataFrame(dev_rows)), ""]

    # --- B: severity ---
    dmg = df[df["_damaged"] == 1].reset_index(drop=True)
    yB, grpB = dmg["_severity"].values, dmg[CASE_COL].astype(str).values
    rowsB = []
    for sname in ("base(5)", "extended(8)"):
        cols = sets[sname]
        for m in ["logreg", "svm_rbf", "knn15", "extra_trees", "random_forest"]:
            P, cl = loco(lambda m=m: clf(m), dmg[cols], yB, grpB)
            pr = cl[P.argmax(1)]
            ok, n = case_vote_acc(grpB, yB, pr)
            rowsB.append(dict(features=sname, approach="classifier", model=m, bal_acc=balanced_accuracy_score(yB, pr),
                              f1_macro=f1_score(yB, pr, average="macro"), mae_mm=np.abs(pr - yB).mean(), cases_right=f"{ok}/{n}"))
            LOG.info("B %-12s %-14s done", sname, m)
        for m in ["ridge", "rf_reg", "gb_reg"]:
            raw = loco(lambda m=m: reg(m), dmg[cols], yB, grpB, proba=False)
            pr = np.clip(np.rint(raw), 1, 3).astype(int)
            ok, n = case_vote_acc(grpB, yB, pr)
            rowsB.append(dict(features=sname, approach="ordinal regression, rounded", model=m, bal_acc=balanced_accuracy_score(yB, pr),
                              f1_macro=f1_score(yB, pr, average="macro"), mae_mm=np.abs(pr - yB).mean(), cases_right=f"{ok}/{n}"))
            LOG.info("B %-12s %-14s done", sname, m)
    B = pd.DataFrame(rowsB).sort_values("f1_macro", ascending=False)
    md += ["## 2. Model B: severity (damaged recordings only)\n", md_table(B), ""]
    md += ["Compare these rows with the saved model's scores in training_report.md (they are produced by the same folds).", "",
           "Selecting the winner here uses the same folds that score it, so its score is optimistic. Report it as such.", ""]
    text = "\n".join(md)
    args.report_dir.mkdir(parents=True, exist_ok=True)
    (args.report_dir / "experiments.md").write_text(text)
    print(text)
    LOG.info("Total %.0fs; saved %s", time.time() - t0, args.report_dir / "experiments.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())