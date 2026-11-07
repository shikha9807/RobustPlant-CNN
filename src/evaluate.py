"""Multi-metric evaluation: in-domain (PlantVillage test) and cross-domain
(PlantDoc) with NO retraining or fine-tuning on PlantDoc.

Usage:
    python -m src.evaluate --run-name robustplant_b0

Outputs (results/<run>/):
    metrics.json                 all metrics + domain-gap summary
    per_class_<domain>.csv       precision / recall / F1 / support per class
    logits_<domain>.npz          logits, labels, paths (for calibration.py)
    visualizations/<run>_confusion_plantdoc.png
"""
from __future__ import annotations

import argparse
import json
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import tensorflow as tf  # noqa: E402
from sklearn.metrics import (confusion_matrix,  # noqa: E402
                             precision_recall_fscore_support)

from src import config  # noqa: E402
from src.dataset_loader import build_dataset, load_or_build_splits  # noqa: E402
from src.model import (WEIGHTS_FILE, get_logits_model,  # noqa: E402
                       load_trained_model, model_dir)


# --------------------------------------------------------------------------- #
# Metric helpers (pure NumPy)
# --------------------------------------------------------------------------- #
def softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray,
                               n_bins: int = config.ECE_BINS) -> float:
    """Top-label ECE with equal-width confidence bins."""
    conf = probs.max(axis=1)
    correct = (probs.argmax(axis=1) == labels).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        in_bin = (conf > lo) & (conf <= hi)
        if in_bin.any():
            ece += in_bin.mean() * abs(correct[in_bin].mean() - conf[in_bin].mean())
    return float(ece)


def negative_log_likelihood(probs: np.ndarray, labels: np.ndarray) -> float:
    p = np.clip(probs[np.arange(len(labels)), labels], 1e-12, 1.0)
    return float(-np.log(p).mean())


def restrict_to_classes(logits: np.ndarray, allowed: list[int]) -> np.ndarray:
    """Mask logits of classes that cannot be ground truth in this domain."""
    masked = np.full_like(logits, -1e9)
    masked[:, allowed] = logits[:, allowed]
    return masked


def compute_metrics(logits: np.ndarray, labels: np.ndarray) -> dict:
    probs = softmax(logits.astype(np.float64))
    pred = probs.argmax(axis=1)
    present = np.unique(labels)
    prec, rec, f1, _ = precision_recall_fscore_support(
        labels, pred, labels=present, average="macro", zero_division=0)
    _, _, wf1, _ = precision_recall_fscore_support(
        labels, pred, labels=present, average="weighted", zero_division=0)
    top3 = np.argsort(-probs, axis=1)[:, :3]
    healthy = np.array([config.is_healthy(config.IDX_TO_CLASS[i])
                        for i in range(config.NUM_CLASSES)])
    return {
        "n_samples": int(len(labels)),
        "n_classes_in_ground_truth": int(len(present)),
        "accuracy": float((pred == labels).mean()),
        "top3_accuracy": float((top3 == labels[:, None]).any(axis=1).mean()),
        "macro_precision": float(prec),
        "macro_recall": float(rec),
        "macro_f1": float(f1),
        "weighted_f1": float(wf1),
        "healthy_vs_diseased_accuracy": float(
            (healthy[pred] == healthy[labels]).mean()),
        "ece": expected_calibration_error(probs, labels),
        "nll": negative_log_likelihood(probs, labels),
        "mean_confidence": float(probs.max(axis=1).mean()),
    }


def per_class_table(logits: np.ndarray, labels: np.ndarray) -> pd.DataFrame:
    pred = logits.argmax(axis=1)
    present = np.unique(labels)
    prec, rec, f1, support = precision_recall_fscore_support(
        labels, pred, labels=present, zero_division=0)
    return pd.DataFrame({
        "class": [config.IDX_TO_CLASS[i] for i in present],
        "precision": prec, "recall": rec, "f1": f1, "support": support})


def plot_confusion(logits, labels, path, title) -> None:
    pred = logits.argmax(axis=1)
    present = np.unique(labels)
    cm = confusion_matrix(labels, pred, labels=present).astype(float)
    cm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    names = [config.IDX_TO_CLASS[i].replace("___", ": ")[:38] for i in present]
    size = max(8, 0.42 * len(present))
    fig, ax = plt.subplots(figsize=(size + 3, size + 1))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(present)), names, rotation=90, fontsize=7)
    ax.set_yticks(range(len(present)), names, fontsize=7)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.04, label="Row-normalised rate")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Inference and latency
# --------------------------------------------------------------------------- #
def predict_logits(logits_model, df: pd.DataFrame, batch_size: int) -> np.ndarray:
    ds = build_dataset(df, training=False, batch_size=batch_size)
    logits = logits_model.predict(ds, verbose=1)
    if len(logits) != len(df):
        raise RuntimeError(f"Got {len(logits)} predictions for {len(df)} images")
    return logits


def measure_latency(logits_model, df: pd.DataFrame, samples: int,
                    batch_size: int, warmup: int = 10) -> dict:
    """Latency at batch size 1 and throughput at `batch_size` (steady state)."""
    sub = df.head(samples + warmup)
    single = [x for x, _ in build_dataset(sub, False, batch_size=1)]
    for x in single[:warmup]:
        logits_model(x, training=False).numpy()
    times = []
    for x in single[warmup:]:
        t0 = time.perf_counter()
        logits_model(x, training=False).numpy()
        times.append((time.perf_counter() - t0) * 1000.0)
    batches = [x for x, _ in build_dataset(df.head(batch_size * 6), False,
                                           batch_size=batch_size)]
    logits_model(batches[0], training=False).numpy()
    t0 = time.perf_counter()
    n_images = 0
    for x in batches[1:]:
        logits_model(x, training=False).numpy()
        n_images += int(x.shape[0])
    elapsed = time.perf_counter() - t0
    return {
        "device": "GPU" if tf.config.list_physical_devices("GPU") else "CPU",
        "batch1_latency_ms_mean": float(np.mean(times)),
        "batch1_latency_ms_std": float(np.std(times)),
        "batch1_latency_ms_p50": float(np.percentile(times, 50)),
        "batch1_latency_ms_p95": float(np.percentile(times, 95)),
        "batch1_samples": len(times),
        "throughput_images_per_s": float(n_images / elapsed) if elapsed else None,
        "throughput_batch_size": batch_size,
    }


def evaluate_domain(name, logits_model, df, batch_size, out_dir,
                    shared_only: list[int] | None = None) -> tuple:
    labels = df["label_idx"].to_numpy(dtype=np.int64)
    logits = predict_logits(logits_model, df, batch_size)
    np.savez_compressed(out_dir / f"logits_{name}.npz", logits=logits,
                        labels=labels, paths=df["rel_path"].to_numpy())
    result = {"all_38_outputs": compute_metrics(logits, labels)}
    per_class_table(logits, labels).to_csv(
        out_dir / f"per_class_{name}.csv", index=False)
    if shared_only is not None:
        masked = restrict_to_classes(logits, shared_only)
        result["shared_classes_only"] = compute_metrics(masked, labels)
        per_class_table(masked, labels).to_csv(
            out_dir / f"per_class_{name}_shared_only.csv", index=False)
    return result, logits, labels


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--run-name", required=True)
    p.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    p.add_argument("--latency-samples", type=int, default=100)
    p.add_argument("--skip-latency", action="store_true")
    args = p.parse_args()

    config.set_global_seed()
    out_dir = config.RESULTS_DIR / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    if not config.PD_MANIFEST.exists():
        raise SystemExit("PlantDoc manifest missing. Run: python -m "
                         "src.data_preprocessing --dataset plantdoc")
    model, cfg = load_trained_model(args.run_name)
    logits_model = get_logits_model(model)

    pv_test = load_or_build_splits().query("split == 'test'").reset_index(drop=True)
    pd_df = pd.read_csv(config.PD_MANIFEST)
    shared = sorted(int(i) for i in pd_df["label_idx"].unique())

    metrics = {"run": args.run_name, "seed": config.SEED,
               "tensorflow": tf.__version__,
               "model_config": {k: v for k, v in cfg.__dict__.items()
                                if k != "class_names"}}
    print("\n--- In-domain: PlantVillage test split ---")
    metrics["plantvillage_test"], _, _ = evaluate_domain(
        "plantvillage_test", logits_model, pv_test, args.batch_size, out_dir)
    print("\n--- Cross-domain: PlantDoc (zero-shot, no retraining) ---")
    metrics["plantdoc"], pd_logits, pd_labels = evaluate_domain(
        "plantdoc", logits_model, pd_df, args.batch_size, out_dir,
        shared_only=shared)

    in_dom = metrics["plantvillage_test"]["all_38_outputs"]
    cross = metrics["plantdoc"]["shared_classes_only"]
    metrics["domain_gap"] = {
        "accuracy_drop": in_dom["accuracy"] - cross["accuracy"],
        "macro_f1_drop": in_dom["macro_f1"] - cross["macro_f1"],
        "ece_increase": cross["ece"] - in_dom["ece"],
        "note": "PlantDoc compared using predictions restricted to the "
                f"{len(shared)} shared classes.",
    }
    if not args.skip_latency:
        metrics["inference"] = measure_latency(
            logits_model, pv_test, args.latency_samples, args.batch_size)
    weights_path = model_dir(args.run_name) / WEIGHTS_FILE
    metrics["model_size"] = {
        "parameters": int(model.count_params()),
        "weights_file_mb": round(weights_path.stat().st_size / 2**20, 2)}

    plot_confusion(restrict_to_classes(pd_logits, shared), pd_labels,
                   config.VIS_DIR / f"{args.run_name}_confusion_plantdoc.png",
                   f"{args.run_name}: PlantDoc confusion (shared classes)")
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps({k: metrics[k] for k in
                      ("plantvillage_test", "plantdoc", "domain_gap")}, indent=2))
    print(f"\nSaved to {out_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
