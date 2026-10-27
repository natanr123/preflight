"""Measure what cheap models can predict from what is known before posting.

    uv run python -m preflight.baseline [--all-languages]

**Why.** Fine-tuning a 30B model is worth its GPU bill only if it beats these
numbers, and if none of them beats a coin flip the plan needs rethinking before
any money is spent. What a Reel looks like, probably the strongest signal, is
not here yet: the crawl skips thumbnails, and the fine-tune adds them.

**Scores.** ROC AUC, where 0.5 is a coin flip and 1.0 is perfect, for two
questions: did the Reel beat its account's median, and did it reach twice it.

**Splits.** *New accounts*: five folds grouped by account, so every test Reel is
from an account the model never saw. *Newer Reels*: each account's newest 30%
is held out, which is what a creator who connects their own account would get.
The account's recent streak (how its last few Reels did) is shown for
reference: it is known before posting but says nothing about the Reel itself.
"""
import argparse
import json
import sys
from collections import defaultdict

import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from preflight.dataset import OUT as DATASET

RESULTS = DATASET.parent / "baseline.json"
#: The laptop is shared: left alone, scikit-learn took all 24 cores (load 43).
THREADS = 4
NUMERIC = ("caption_chars", "words", "hashtags", "lines", "emojis", "question", "cta",
           "hour_utc", "weekday", "gap_hours", "followers", "media_count")
#: Counts that spread over orders of magnitude; logs keep the linear model sane.
LOGGED = {"caption_chars", "words", "hashtags", "lines", "emojis", "gap_hours", "followers", "media_count"}
TARGETS = {"beats median": lambda row: row["beats_median"],
           "2x or more": lambda row: row["band"] == "outperform"}
HELD_OUT = 0.3


def numeric(rows: list[dict]) -> np.ndarray:
    def value(row: dict, name: str) -> float:
        if row[name] is None:
            return np.nan
        return np.log1p(float(row[name])) if name in LOGGED else float(row[name])
    return np.array([[value(row, name) for name in NUMERIC] for row in rows])


def captions(rows: list[dict]) -> list[str]:
    return [row["caption"] for row in rows]


def vectorizer() -> TfidfVectorizer:
    # Hashtags stay whole: "#giveaway" and "giveaway" are different signals.
    return TfidfVectorizer(ngram_range=(1, 2), min_df=3, sublinear_tf=True,
                           token_pattern=r"(?u)#?\b\w\w+\b")


def caption_and_timing(train: list[dict], test: list[dict], y: np.ndarray) -> np.ndarray:
    model = HistGradientBoostingClassifier(learning_rate=0.05, max_leaf_nodes=15,
                                           l2_regularization=1.0, random_state=0)
    return model.fit(numeric(train), y).predict_proba(numeric(test))[:, 1]


def caption_words(train: list[dict], test: list[dict], y: np.ndarray) -> np.ndarray:
    words = vectorizer()
    model = LogisticRegression(max_iter=2000).fit(words.fit_transform(captions(train)), y)
    return model.predict_proba(words.transform(captions(test)))[:, 1]


def both(train: list[dict], test: list[dict], y: np.ndarray) -> np.ndarray:
    words, scaler = vectorizer(), StandardScaler()
    fill = np.nanmedian(numeric(train), axis=0)

    def dense(rows: list[dict]) -> np.ndarray:
        values = numeric(rows)
        return np.where(np.isnan(values), fill, values)

    train_matrix = hstack([words.fit_transform(captions(train)),
                           csr_matrix(scaler.fit_transform(dense(train)))]).tocsr()
    test_matrix = hstack([words.transform(captions(test)),
                          csr_matrix(scaler.transform(dense(test)))]).tocsr()
    return LogisticRegression(max_iter=2000).fit(train_matrix, y).predict_proba(test_matrix)[:, 1]


MODELS = {"caption + timing": caption_and_timing, "caption words": caption_words, "both": both}


def new_accounts(rows: list[dict], model, target) -> list[float]:
    y = np.array([target(row) for row in rows])
    groups = [row["username"] for row in rows]
    scores = []
    for train_index, test_index in GroupKFold(n_splits=5).split(rows, y, groups):
        train, test = [rows[i] for i in train_index], [rows[i] for i in test_index]
        scores.append(roc_auc_score(y[test_index], model(train, test, y[train_index])))
    return scores


def newer_reels(rows: list[dict], model, target) -> float:
    by_account = defaultdict(list)
    for row in rows:
        by_account[row["username"]].append(row)
    train, test = [], []
    for reels in by_account.values():
        reels.sort(key=lambda row: row["timestamp"])
        cut = round(len(reels) * (1 - HELD_OUT))
        train += reels[:cut]
        test += reels[cut:]
    y_train = np.array([target(row) for row in train])
    return roc_auc_score([target(row) for row in test], model(train, test, y_train))


def streak(rows: list[dict], target) -> float:
    known = [row for row in rows if row["streak"] is not None]
    return roc_auc_score([target(row) for row in known], [row["streak"] for row in known])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--all-languages", action="store_true", help="keep non-English captions too")
    args = parser.parse_args()

    rows = [json.loads(line) for line in DATASET.open()]
    if not args.all_languages:
        rows = [row for row in rows if row["language"] in ("en", "und")]
    print(f"{len(rows)} Reels from {len({row['username'] for row in rows})} accounts; "
          + "; ".join(f"{name}: {np.mean([target(row) for row in rows]):.0%}" for name, target in TARGETS.items()))

    results = {}
    with threadpool_limits(limits=THREADS):
        for name, model in MODELS.items():
            results[name] = {}
            for target_name, target in TARGETS.items():
                folds = new_accounts(rows, model, target)
                results[name][target_name] = {"new_accounts": float(np.mean(folds)), "sd": float(np.std(folds)),
                                              "newer_reels": newer_reels(rows, model, target)}
    results["recent streak (reference)"] = {target_name: {"all": streak(rows, target)}
                                            for target_name, target in TARGETS.items()}
    RESULTS.write_text(json.dumps(results, indent=2))

    print(f"\n{'ROC AUC, 0.5 = coin flip':<28}" + "".join(f"{name:<24}" for name in TARGETS))
    print(f"{'':<28}" + f"{'new accounts  newer':<24}" * len(TARGETS))
    for name in MODELS:
        print(f"{name:<28}" + "".join(
            f"{cell['new_accounts']:.3f} ±{cell['sd']:.2f}   {cell['newer_reels']:.3f}    "
            for cell in results[name].values()))
    print(f"{'recent streak (reference)':<28}" + "".join(
        f"{cell['all']:.3f}{'':<19}" for cell in results["recent streak (reference)"].values()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
