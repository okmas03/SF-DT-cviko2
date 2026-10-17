"""
10-fold stratifikovaná cross-validácia pre Gaussian Naive Bayes a kNN.

Pre každý variant dát (bez/so riedkymi stĺpcami) a každú sadu príznakov:
  * v každom folde sa MinMaxScaler fituje LEN na trénovacích dátach,
  * GNB sa natrénuje a vyhodnotí,
  * kNN sa vyhodnotí pre viacero hodnôt k naraz (susedia sa hľadajú raz pre
    najväčšie k, menšie k sú jeho prefixom -> rovnaký výsledok ako
    KNeighborsClassifier, len rýchlejšie),
  * confusion matice sa sčítajú cez všetky foldy.

Výstupy idú do results/.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             confusion_matrix, f1_score, precision_score,
                             recall_score)
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler

from preprocessing import knn_features, load_raw, preprocess

RESULTS = Path(__file__).resolve().parent.parent / "results"
N_FOLDS = 10
K_VALUES = [1, 5, 11, 21, 31, 51, 75, 101]
SEED = 42

VARIANTS = {
    "A": dict(drop_sparse=True,  name="A – bez Sunshine/Evaporation/Cloud"),
    "B": dict(drop_sparse=False, name="B – všetky stĺpce"),
}


def metrics(y_true, y_pred) -> dict:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
    }


def run_variant(X: pd.DataFrame, y: pd.Series):
    """Vráti {model_name: (zoznam metrík po foldoch, sčítaná confusion matrix)}."""
    feature_sets = {"all": list(X.columns), "selected": knn_features(X)}
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    fold_metrics, cms = {}, {}

    def record(model, y_true, y_pred):
        fold_metrics.setdefault(model, []).append(metrics(y_true, y_pred))
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        cms[model] = cms.get(model, 0) + cm

    y_np = y.to_numpy()
    for fold, (tr, te) in enumerate(skf.split(X, y_np), start=1):
        y_tr, y_te = y_np[tr], y_np[te]

        # Baseline: vždy "nebude pršať"
        record("baseline (vždy No)", y_te, np.zeros_like(y_te))

        # Gaussian NB (škálovanie nemá vplyv, robíme ho kvôli jednotnosti)
        scaler = MinMaxScaler().fit(X.iloc[tr])
        gnb = GaussianNB().fit(scaler.transform(X.iloc[tr]), y_tr)
        record("GNB (všetky príznaky)", y_te, gnb.predict(scaler.transform(X.iloc[te])))

        # kNN pre obe sady príznakov a všetky k
        for fs_name, cols in feature_sets.items():
            scaler = MinMaxScaler().fit(X.iloc[tr][cols])
            Xtr = scaler.transform(X.iloc[tr][cols])
            Xte = scaler.transform(X.iloc[te][cols])
            nn = NearestNeighbors(n_neighbors=max(K_VALUES), algorithm="brute", n_jobs=-1)
            idx = nn.fit(Xtr).kneighbors(Xte, return_distance=False)
            neigh_labels = y_tr[idx]  # (n_test, max_k)
            for k in K_VALUES:
                pred = (neigh_labels[:, :k].mean(axis=1) > 0.5).astype(int)
                record(f"kNN k={k} ({fs_name})", y_te, pred)

        print(f"  fold {fold}/{N_FOLDS} hotový")

    return fold_metrics, cms


def summarize(fold_metrics) -> pd.DataFrame:
    rows = []
    for model, folds in fold_metrics.items():
        df = pd.DataFrame(folds)
        row = {"model": model}
        for m in df.columns:
            row[m] = df[m].mean()
            row[f"{m}_std"] = df[m].std()
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    RESULTS.mkdir(exist_ok=True)
    raw = load_raw()
    all_summaries, all_cms = [], {}

    for key, cfg in VARIANTS.items():
        X, y = preprocess(raw, drop_sparse=cfg["drop_sparse"])
        print(f"Variant {cfg['name']}: {X.shape[0]} riadkov, {X.shape[1]} príznakov")
        fm, cms = run_variant(X, y)
        summary = summarize(fm)
        summary.insert(0, "variant", key)
        summary.insert(1, "n_rows", len(X))
        all_summaries.append(summary)
        for model, cm in cms.items():
            all_cms[(key, model)] = cm

    summary = pd.concat(all_summaries, ignore_index=True)
    summary.to_csv(RESULTS / "metrics.csv", index=False, float_format="%.4f")

    cm_rows = [
        {"variant": v, "model": m, "TN": cm[0, 0], "FP": cm[0, 1], "FN": cm[1, 0], "TP": cm[1, 1]}
        for (v, m), cm in all_cms.items()
    ]
    pd.DataFrame(cm_rows).to_csv(RESULTS / "confusion_matrices.csv", index=False)
    print(f"\nUložené do {RESULTS}/")


if __name__ == "__main__":
    main()