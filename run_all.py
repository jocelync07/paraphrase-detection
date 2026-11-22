"""Train and evaluate every model on the same PAWS subset; write results/.

    python run_all.py                          # 5,000 train / 1,000 test pairs (about 30-40 min on a laptop CPU)
    python run_all.py --train 1000 --test 300  # a quick check

Downloads PAWS and two pre-trained models from the Hugging Face Hub on first run.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from paraphrase import baseline, cross_encoder, retrieval
from paraphrase.data import labels, load_paws
from paraphrase.metrics import classification

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results"


def run_baselines(train, test, seed: int) -> dict:
    from sklearn.model_selection import StratifiedKFold, cross_val_score

    x_train, x_test = baseline.feature_matrix(train), baseline.feature_matrix(test)
    y_train, y_test = np.array(labels(train)), np.array(labels(test))
    out = {}
    for name, model in baseline.models().items():
        folds = StratifiedKFold(5, shuffle=True, random_state=seed)
        cv = cross_val_score(model, x_train, y_train, cv=folds, scoring="accuracy")
        model.fit(x_train, y_train)
        out[name] = {**classification(y_test, model.predict(x_test)),
                     "cv_accuracy_mean": float(cv.mean()), "cv_accuracy_std": float(cv.std())}
        print(f"  {name}: test accuracy {out[name]['accuracy']:.4f} (5-fold CV {cv.mean():.4f} +/- {cv.std():.4f})")
    majority = int(y_train.mean() >= 0.5)
    out["Majority class"] = classification(y_test, np.full_like(y_test, majority))
    return out


def chart(results: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    models = {
        "Majority class": results["baselines"]["Majority class"],
        "Naive Bayes": results["baselines"]["Naive Bayes"],
        "Logistic regression": results["baselines"]["Logistic regression"],
        "Sentence-BERT\n(pre-trained)": results["sentence_bert"]["before"],
        "Sentence-BERT\n(fine-tuned)": results["sentence_bert"]["after"],
        "DistilBERT\ncross-encoder": results["distilbert"],
    }
    metrics = ["accuracy", "f1", "macro_f1"]
    fig, ax = plt.subplots(figsize=(10, 4.8))
    width = 0.8 / len(metrics)
    for i, metric in enumerate(metrics):
        values = [m[metric] for m in models.values()]
        bars = ax.bar([x + i * width for x in range(len(models))], values, width,
                      label={"f1": "F1 (paraphrase)", "macro_f1": "Macro-F1"}.get(metric, "Accuracy"))
        ax.bar_label(bars, fmt="%.2f", fontsize=7)
    ax.set_xticks([x + width for x in range(len(models))], list(models), fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_title("PAWS test set: models compared")
    ax.legend(loc="upper left", ncol=3, frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "comparison.png", dpi=150)
    plt.close(fig)


def write_markdown(results: dict) -> None:
    rows = [("Majority class", results["baselines"]["Majority class"]),
            ("Naive Bayes (8 overlap features)", results["baselines"]["Naive Bayes"]),
            ("Logistic regression (same features)", results["baselines"]["Logistic regression"]),
            ("Sentence-BERT, pre-trained (threshold)", results["sentence_bert"]["before"]),
            ("Sentence-BERT, fine-tuned (threshold)", results["sentence_bert"]["after"]),
            ("DistilBERT cross-encoder, fine-tuned", results["distilbert"])]
    cfg = results["config"]
    lines = [f"PAWS labeled_final: {cfg['train']:,} training pairs, {cfg['test']:,} test pairs (seed {cfg['seed']}).", "",
             "| Model | Accuracy | Precision | Recall | F1 (paraphrase) | Macro-F1 |", "|---|---|---|---|---|---|"]
    for name, m in rows:
        lines.append(f"| {name} | {m['accuracy']:.3f} | {m['precision']:.3f} | {m['recall']:.3f} | "
                     f"{m['f1']:.3f} | {m['macro_f1']:.3f} |")
    b, a = results["sentence_bert"]["before"], results["sentence_bert"]["after"]
    lines += ["", "Retrieval (query = sentence 1 of each test paraphrase; candidates = all test sentence 2s):", "",
              "| Sentence-BERT | Recall@1 | Recall@5 | MRR | Similarity gap (para - non-para) |", "|---|---|---|---|---|",
              f"| Pre-trained | {b['recall@1']:.3f} | {b['recall@5']:.3f} | {b['mrr']:.3f} | {b['similarity_gap']:.3f} |",
              f"| Fine-tuned | {a['recall@1']:.3f} | {a['recall@5']:.3f} | {a['mrr']:.3f} | {a['similarity_gap']:.3f} |",
              "", "DistilBERT training:", "", "| Epoch | Train loss | Test accuracy | Minutes |", "|---|---|---|---|"]
    for h in results["distilbert"]["history"]:
        lines.append(f"| {h['epoch']} | {h['train_loss']:.4f} | {h['test_accuracy']:.3f} | {h['minutes']:.1f} |")
    (OUT / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train", type=int, default=5000)
    parser.add_argument("--test", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sbert-epochs", type=int, default=1)
    parser.add_argument("--bert-epochs", type=int, default=3)
    args = parser.parse_args()
    OUT.mkdir(exist_ok=True)

    began = time.time()
    train, test = load_paws(args.train, args.test, args.seed)
    print(f"Loaded {len(train)} train / {len(test)} test pairs; "
          f"paraphrase share {np.mean(labels(train)):.3f} / {np.mean(labels(test)):.3f}")

    results = {"config": {"train": len(train), "test": len(test), "seed": args.seed,
                          "sbert_epochs": args.sbert_epochs, "bert_epochs": args.bert_epochs}}
    print("1. Overlap-feature baselines")
    results["baselines"] = run_baselines(train, test, args.seed)
    print("2. Sentence-BERT (Siamese)")
    results["sentence_bert"] = retrieval.run(train, test, args.sbert_epochs)
    print("   before:", {k: round(v, 3) for k, v in results["sentence_bert"]["before"].items()})
    print("   after: ", {k: round(v, 3) for k, v in results["sentence_bert"]["after"].items()})
    print("3. DistilBERT cross-encoder")
    results["distilbert"] = cross_encoder.run(train, test, args.bert_epochs, seed=args.seed)

    (OUT / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    write_markdown(results)
    chart(results)
    print(f"Done in {(time.time() - began) / 60:.1f} min. Wrote results/results.md, results.json, comparison.png")


if __name__ == "__main__":
    main()
