"""Train, validate and package the Data Pulse ticket classifier.

Run from the project root:

    python -m training.train                 # full run: experiments + final model
    python -m training.train --skip-cv       # faster: skip the in-distribution cross-validation columns

What it does, in order:
  1. EXPERIMENT LADDER  - six feature sets, each trained on train.jsonl only and scored on
     validation.jsonl. Validation uses phrasings that never occur in train, so this column is
     our estimate of how the model copes with new wording. (Cross-validation inside train is
     also reported: it is ~98% for every model and is therefore useless for model selection.)
  2. HEADS              - with the winning feature set: category, urgency and secondary-category
     classifiers; temperature (confidence calibration), urgency threshold, secondary margin and
     human-review threshold are all tuned on validation.
  3. FINAL MODEL        - the same recipe refitted on train + validation (4,800 tickets) and saved to
     models/<version>/ together with metrics.json and model_card.json.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import platform
import random
import sys
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from scipy.optimize import minimize_scalar
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.features import Featurizer, normalize  # noqa: E402
from app.model import CATEGORIES, NONE, softmax  # noqa: E402
from training.extra_examples import extra_rows  # noqa: E402

warnings.filterwarnings("ignore")
VERSION = "datapulse-1.0.0"
SEED = 42
C_SVM = 0.5


# ----------------------------------------------------------------------------- data
def prepare(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["x"] = [normalize(s, t) for s, t in zip(df.subject, df.text)]
    df["secondary"] = df.secondary_category.fillna(NONE)
    return df


def load(name: str) -> pd.DataFrame:
    return prepare(pd.read_json(ROOT / "data" / f"{name}.jsonl", lines=True))


def with_extra(df: pd.DataFrame) -> pd.DataFrame:
    """Append our hand-written spam examples (see extra_examples.py). Fitting only, never scoring."""
    return pd.concat([df, prepare(pd.DataFrame(extra_rows()))], ignore_index=True)


# ----------------------------------------------------------------------------- metrics helpers
def ece(conf: np.ndarray, correct: np.ndarray, bins: int = 10):
    """Expected calibration error + the reliability table behind it."""
    edges = np.linspace(0, 1, bins + 1)
    total, rows = 0.0, []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.sum() == 0:
            continue
        total += m.mean() * abs(correct[m].mean() - conf[m].mean())
        rows.append({"bin": f"{lo:.1f}-{hi:.1f}", "n": int(m.sum()), "confidence": round(float(conf[m].mean()), 3),
                     "accuracy": round(float(correct[m].mean()), 3)})
    return round(float(total), 4), rows


def by_group(df: pd.DataFrame, col: str, pred: np.ndarray) -> dict:
    ok = pd.Series(pred == df.category.values, index=df.index)
    return {str(k): {"n": int(len(v)), "accuracy": round(float(v.mean()), 4)} for k, v in ok.groupby(df[col])}


def add_noise(text: str, rng: random.Random) -> str:
    """Cheap robustness probe: drop ~4% of characters and squash casing/punctuation."""
    chars = [ch for ch in text if rng.random() > 0.04]
    return "".join(chars).replace(".", " ").replace(",", " ")


# ----------------------------------------------------------------------------- experiment ladder
LADDER = [
    ("E1", "Baseline: word TF-IDF + Logistic Regression",
     dict(use_char=False, use_rom=False, use_lexicon=False, use_masked_lexicon=False), "lr"),
    ("E2", "Word TF-IDF + linear SVM",
     dict(use_char=False, use_rom=False, use_lexicon=False, use_masked_lexicon=False), "svm"),
    ("E3", "+ character n-grams (typo / spelling robust)",
     dict(use_rom=False, use_lexicon=False, use_masked_lexicon=False), "svm"),
    ("E4", "+ romanised character n-grams (one script for Sinhala and Singlish)",
     dict(use_lexicon=False, use_masked_lexicon=False), "svm"),
    ("E5", "+ domain concept lexicon", dict(use_masked_lexicon=False), "svm"),
    ("E6", "+ boilerplate-masked lexicon (FINAL)", dict(), "svm"),
]


def make_clf(kind: str):
    if kind == "lr":
        return LogisticRegression(C=10, max_iter=3000)
    return LinearSVC(C=C_SVM, random_state=SEED)


def run_ladder(train: pd.DataFrame, val: pd.DataFrame, skip_cv: bool) -> list[dict]:
    rows = []
    for code, label, kwargs, kind in LADDER:
        t0 = time.time()
        fz = Featurizer(**kwargs)
        Xtr = fz.fit_transform(list(train.x), train.category.values)
        clf = make_clf(kind).fit(Xtr, train.category.values)
        pred = clf.predict(fz.transform(list(val.x)))
        row = {
            "id": code, "name": label,
            "val_accuracy": round(accuracy_score(val.category, pred), 4),
            "val_macro_f1": round(f1_score(val.category, pred, average="macro"), 4),
            "val_by_language": {k: v["accuracy"] for k, v in by_group(val, "language", pred).items()},
            "n_features": int(Xtr.shape[1]),
        }
        if not skip_cv:
            scores = []
            for a, b in StratifiedKFold(3, shuffle=True, random_state=SEED).split(train.x, train.category):
                f2 = Featurizer(**kwargs)
                Xa = f2.fit_transform(list(train.x.values[a]), train.category.values[a])
                c2 = make_clf(kind).fit(Xa, train.category.values[a])
                scores.append(accuracy_score(train.category.values[b], c2.predict(f2.transform(list(train.x.values[b])))))
            row["train_cv_accuracy"] = round(float(np.mean(scores)), 4)
        rows.append(row)
        print(f"  {code} {label:72s} val acc {row['val_accuracy']:.4f}  macro-F1 {row['val_macro_f1']:.4f}"
              f"  train-CV {row.get('train_cv_accuracy', float('nan')):.4f}  [{time.time() - t0:.0f}s]", flush=True)
    return rows


# ----------------------------------------------------------------------------- heads + tuning
def fit_heads(df: pd.DataFrame):
    fz = Featurizer()
    X = fz.fit_transform(list(df.x), df.category.values)
    clf_cat = LinearSVC(C=C_SVM, random_state=SEED).fit(X, df.category.values)
    clf_urg = LinearSVC(C=C_SVM, class_weight="balanced", random_state=SEED).fit(X, df.is_urgent.values.astype(int))
    clf_sec = LinearSVC(C=C_SVM, class_weight="balanced", random_state=SEED).fit(X, df.secondary.values)
    return fz, clf_cat, clf_urg, clf_sec


def secondary_decode(scores: np.ndarray, classes: list[str], primary: np.ndarray, margin: float) -> np.ndarray:
    out = []
    none_idx = classes.index(NONE)
    for i in range(scores.shape[0]):
        s = scores[i].copy()
        none_score = s[none_idx]
        s[none_idx] = -np.inf
        if primary[i] in classes:
            s[classes.index(primary[i])] = -np.inf
        j = int(np.argmax(s))
        label = classes[j] if s[j] - none_score > margin else NONE
        out.append(NONE if primary[i] == "spam_irrelevant" else label)
    return np.array(out)


def secondary_scores(true: np.ndarray, pred: np.ndarray) -> dict:
    tp = int(((pred == true) & (true != NONE)).sum())
    n_pred, n_true = int((pred != NONE).sum()), int((true != NONE).sum())
    p = tp / n_pred if n_pred else 0.0
    r = tp / n_true if n_true else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return {"exact_match_accuracy": round(float((pred == true).mean()), 4), "precision": round(p, 4),
            "recall": round(r, 4), "f1": round(f, 4), "n_true_secondary": n_true, "n_predicted_secondary": n_pred}


def tune_and_evaluate(train: pd.DataFrame, val: pd.DataFrame) -> tuple[dict, dict]:
    fz, clf_cat, clf_urg, clf_sec = fit_heads(train)
    Xv = fz.transform(list(val.x))
    yv = val.category.values
    scores = clf_cat.decision_function(Xv)
    classes = list(clf_cat.classes_)
    y_idx = np.array([classes.index(c) for c in yv])

    # --- confidence calibration: one temperature for softmax(margins / T), fitted by NLL on validation
    def nll(t):
        p = softmax(scores, t)
        return -np.log(p[np.arange(len(y_idx)), y_idx] + 1e-12).mean()

    temperature = float(minimize_scalar(nll, bounds=(0.02, 5.0), method="bounded").x)
    probs = softmax(scores, temperature)
    pred = np.array(classes)[probs.argmax(1)]
    conf = probs.max(1)
    correct = (pred == yv).astype(float)
    ece_raw, _ = ece(softmax(scores, 1.0).max(1), correct)
    ece_cal, reliability = ece(conf, correct)

    # --- urgency threshold: maximise F1 on validation
    us = clf_urg.decision_function(Xv)
    best_thr, best_f1 = 0.0, -1.0
    for thr in np.arange(-0.6, 0.61, 0.05):
        f = f1_score(val.is_urgent.values, us > thr)
        if f > best_f1 + 1e-9:
            best_thr, best_f1 = float(round(thr, 2)), f
    urg_pred = (us > best_thr) & (pred != "spam_irrelevant")
    up, ur, uf, _ = precision_recall_fscore_support(val.is_urgent.values, urg_pred, average="binary")

    # --- secondary margin: maximise exact-label F1 on validation
    ss = clf_sec.decision_function(Xv)
    sec_classes = list(clf_sec.classes_)
    best_margin, best = 0.0, {"f1": -1.0}
    for margin in np.arange(-1.0, 1.01, 0.05):
        sc = secondary_scores(val.secondary.values, secondary_decode(ss, sec_classes, pred, margin))
        if sc["f1"] > best["f1"] + 1e-9:
            best_margin, best = float(round(margin, 2)), sc

    # --- human-review threshold: highest coverage whose auto-routed tickets are >= 95% accurate
    review_rows, review_threshold = [], 0.5
    for thr in np.arange(0.30, 0.91, 0.05):
        keep = conf >= thr
        review_rows.append({
            "threshold": round(float(thr), 2), "auto_routed_share": round(float(keep.mean()), 3),
            "auto_routed_accuracy": round(float(correct[keep].mean()), 3) if keep.any() else None,
            "flagged_accuracy": round(float(correct[~keep].mean()), 3) if (~keep).any() else None,
        })
    for row in review_rows:
        if row["auto_routed_accuracy"] is not None and row["auto_routed_accuracy"] >= 0.95:
            review_threshold = row["threshold"]
            break

    # --- robustness probe: noisy copy of validation
    rng = random.Random(SEED)
    noisy = [normalize("", add_noise(x, rng)) for x in val.x]
    noisy_pred = clf_cat.predict(fz.transform(noisy))

    p_, r_, f_, s_ = precision_recall_fscore_support(yv, pred, labels=CATEGORIES, zero_division=0)
    metrics = {
        "protocol": "Fitted on train.jsonl only (4,000 tickets + 30 hand-written spam examples), scored on validation.jsonl (800 tickets with unseen phrasing).",
        "category": {
            "accuracy": round(accuracy_score(yv, pred), 4),
            "macro_f1": round(f1_score(yv, pred, average="macro"), 4),
            "weighted_f1": round(f1_score(yv, pred, average="weighted"), 4),
            "per_class": [{"category": c, "precision": round(float(p_[i]), 3), "recall": round(float(r_[i]), 3),
                           "f1": round(float(f_[i]), 3), "support": int(s_[i])} for i, c in enumerate(CATEGORIES)],
            "confusion_matrix": {"labels": CATEGORIES, "rows": confusion_matrix(yv, pred, labels=CATEGORIES).tolist()},
            "by_language": by_group(val, "language", pred),
            "by_channel": by_group(val, "channel", pred),
            "noisy_validation_accuracy": round(accuracy_score(yv, noisy_pred), 4),
        },
        "urgency": {"threshold": best_thr, "precision": round(float(up), 4), "recall": round(float(ur), 4),
                    "f1": round(float(uf), 4), "positives": int(val.is_urgent.sum())},
        "secondary": {"margin": best_margin, **best},
        "calibration": {"temperature": round(temperature, 4), "ece_before": ece_raw, "ece_after": ece_cal,
                        "reliability": reliability},
        "human_review": {"threshold": review_threshold, "table": review_rows},
    }
    tuned = {"temperature": temperature, "urgent_threshold": best_thr, "secondary_margin": best_margin,
             "review_threshold": review_threshold}
    return metrics, tuned


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-cv", action="store_true", help="skip in-distribution cross-validation (faster)")
    args = ap.parse_args()
    np.random.seed(SEED)
    random.seed(SEED)

    train, val = load("train"), load("validation")
    print(f"train {len(train)} tickets, validation {len(val)} tickets\n\n[1/3] experiment ladder (fit on train, score on validation)")
    ladder = run_ladder(train, val, args.skip_cv)

    print("\n[2/3] tuning heads on validation")
    metrics, tuned = tune_and_evaluate(with_extra(train), val)
    c, u, s, k = metrics["category"], metrics["urgency"], metrics["secondary"], metrics["calibration"]
    print(f"  category   acc {c['accuracy']:.4f}  macro-F1 {c['macro_f1']:.4f}  noisy-val acc {c['noisy_validation_accuracy']:.4f}")
    print(f"  urgency    P {u['precision']:.3f}  R {u['recall']:.3f}  F1 {u['f1']:.3f}  (threshold {u['threshold']})")
    print(f"  secondary  exact-match {s['exact_match_accuracy']:.4f}  P {s['precision']:.3f}  R {s['recall']:.3f}  F1 {s['f1']:.3f}  (margin {s['margin']})")
    print(f"  calibration T={k['temperature']}  ECE {k['ece_before']} -> {k['ece_after']}   review threshold {metrics['human_review']['threshold']}")

    print("\n[3/3] final model: refit on train + validation")
    full = with_extra(pd.concat([train, val], ignore_index=True))
    fz, clf_cat, clf_urg, clf_sec = fit_heads(full)
    out_dir = ROOT / "models" / VERSION
    out_dir.mkdir(parents=True, exist_ok=True)
    artifact = {"version": VERSION, "featurizer": fz, "clf_category": clf_cat, "clf_urgent": clf_urg,
                "clf_secondary": clf_sec, **tuned}
    joblib.dump(artifact, out_dir / "model.joblib", compress=3)
    sha = hashlib.sha256((out_dir / "model.joblib").read_bytes()).hexdigest()

    metrics["experiments"] = ladder
    metrics["dataset"] = {
        "train": len(train), "validation": len(val),
        "languages": train.language.value_counts().to_dict(),
        "channels": train.channel.value_counts().to_dict(),
        "categories": train.category.value_counts().to_dict(),
        "urgent_share": round(float(train.is_urgent.mean()), 3),
        "secondary_share": round(float((train.secondary != NONE).mean()), 3),
    }
    metrics["model_version"] = VERSION
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=1, ensure_ascii=False), encoding="utf-8")
    card = {
        "model_version": VERSION,
        "sha256_model_joblib": sha,
        "trained_at_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "algorithm": "Linear SVM (one-vs-rest) x3 heads on TF-IDF word + char n-grams, romanised char n-grams and a domain concept lexicon",
        "trained_on": f"train.jsonl + validation.jsonl + {len(extra_rows())} hand-written spam examples ({len(full)} tickets)",
        "hyperparameters": {"C": C_SVM, "lexicon_weight": fz.lexicon_weight, **{k_: round(v, 4) for k_, v in tuned.items()}},
        "n_features": int(fz.transform(["x"]).shape[1]),
        "n_filler_trigrams": len(fz.fillers),
        "libraries": {"python": platform.python_version(), "scikit-learn": sklearn.__version__, "numpy": np.__version__},
        "validation_metrics_train_only_fit": {"accuracy": c["accuracy"], "macro_f1": c["macro_f1"],
                                              "urgency_f1": u["f1"], "secondary_f1": s["f1"], "ece": k["ece_after"]},
    }
    (out_dir / "model_card.json").write_text(json.dumps(card, indent=1), encoding="utf-8")
    (ROOT / "models" / "CURRENT").write_text(VERSION + "\n", encoding="utf-8")
    size = (out_dir / "model.joblib").stat().st_size / 1e6
    print(f"  saved {out_dir.relative_to(ROOT)}/model.joblib ({size:.1f} MB, sha256 {sha[:12]}...)\ndone.")


if __name__ == "__main__":
    main()
