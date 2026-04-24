"""
generate_chronos_encoder_features.py

Extract Chronos encoder hidden-state embeddings for each distillation window
and write them as per-window feature CSVs for use as student static features.

For each window, the encoder processes the context sequence and we mean-pool
the last hidden states to a fixed-dim vector, then PCA-reduce to `--n-components`
dimensions. Output CSVs have one row per window:
    item_id, enc_0, enc_1, ..., enc_{n-1}
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=FutureWarning)
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))


def _extract_cli_value(flag: str, default: str) -> str:
    if flag in sys.argv:
        idx = sys.argv.index(flag)
        if idx + 1 < len(sys.argv):
            return sys.argv[idx + 1]
    return default


os.environ["CUDA_VISIBLE_DEVICES"] = _extract_cli_value("--gpu-id", "0")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA

from train_tft_chronos2_cascade import (
    WindowSpec,
    build_all_stations,
    build_window_specs,
    collect_env,
    load_shared_window,
    steps_per_hour,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate Chronos encoder embeddings for distillation windows."
    )
    parser.add_argument("--input-dir", type=Path, default=Path("processed_csv/aligned_stations"))
    parser.add_argument("--metadata-path", type=Path, default=Path("processed_csv/shared_timeline_metadata.json"))
    parser.add_argument("--teacher-cache-dir", type=Path, required=True,
                        help="Directory containing window_index_{split}.csv files from generate_chronos_teacher.py")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Output directory for encoder feature CSVs")
    parser.add_argument("--teacher-model-path", default="amazon/chronos-t5-large")
    parser.add_argument("--target", default="WVHT")
    parser.add_argument("--freq", default="10min")
    parser.add_argument("--gpu-id", default="0")
    parser.add_argument("--context-hours", type=int, default=168)
    parser.add_argument("--n-components", type=int, default=8,
                        help="PCA output dimensions for encoder embedding.")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.20)
    parser.add_argument("--test-ratio", type=float, default=0.10)
    parser.add_argument("--limit-stations", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--no-shared-window", action="store_true")
    return parser


def load_window_specs(path: Path) -> list[WindowSpec]:
    df = pd.read_csv(path)
    records: list[WindowSpec] = []
    for row in df.to_dict(orient="records"):
        records.append(WindowSpec(
            item_id=str(row["item_id"]),
            source_item_id=str(row["source_item_id"]),
            split_name=str(row["split_name"]),
            horizon_hours=int(row["horizon_hours"]),
            origin_index=int(row["origin_index"]),
            origin_timestamp=str(row["origin_timestamp"]),
        ))
    return records


def load_chronos_pipeline(model_path: str, device: str):
    """Load Chronos pipeline using the chronos package directly."""
    from chronos import BaseChronosPipeline
    pipeline = BaseChronosPipeline.from_pretrained(
        model_path,
        device_map=device,
        torch_dtype=torch.float32,
    )
    return pipeline


def extract_encoder_embeddings(
    specs: list[WindowSpec],
    frames: dict[str, pd.DataFrame],
    pipeline,
    target: str,
    context_steps: int,
    batch_size: int,
    device: str,
) -> np.ndarray:
    """
    For each window spec, extract mean-pooled encoder hidden states.
    Returns array of shape (len(specs), hidden_dim).
    """
    model = pipeline.model
    # Access underlying T5 model encoder
    # ChronosModel wraps T5ForConditionalGeneration; encoder is at model.model.encoder
    if hasattr(model, "model") and hasattr(model.model, "encoder"):
        encoder = model.model.encoder
    elif hasattr(model, "encoder"):
        encoder = model.encoder
    else:
        raise RuntimeError(f"Cannot find encoder in Chronos model. Attrs: {dir(model)}")

    encoder.eval()
    all_embeddings: list[np.ndarray] = []

    chunk_size = max(1, batch_size)
    total = len(specs)
    print(f"  Extracting encoder embeddings for {total} windows (batch={chunk_size}) ...")

    for start in range(0, total, chunk_size):
        chunk = specs[start: start + chunk_size]
        contexts: list[torch.Tensor] = []

        for spec in chunk:
            source = frames[spec.source_item_id]
            ctx_start = spec.origin_index - context_steps + 1
            ctx_vals = source.iloc[ctx_start: spec.origin_index + 1][target].values.astype(np.float32)
            # Fill NaN
            mask = np.isnan(ctx_vals)
            if mask.any():
                # forward fill then backward fill
                idx = np.where(~mask)[0]
                if len(idx) == 0:
                    ctx_vals = np.zeros_like(ctx_vals)
                else:
                    filled = ctx_vals.copy()
                    filled[:idx[0]] = ctx_vals[idx[0]]
                    filled[idx[-1]+1:] = ctx_vals[idx[-1]]
                    for i in range(len(filled)):
                        if np.isnan(filled[i]):
                            filled[i] = filled[i-1]
                    ctx_vals = filled
            contexts.append(torch.tensor(ctx_vals, dtype=torch.float32))

        # Tokenize using Chronos pipeline's tokenizer
        # pipeline.tokenizer expects (batch, seq) tensors
        # Keep on CPU for tokenization (tokenizer bucketize runs on CPU)
        context_tensor = torch.stack(contexts)  # (B, T) on CPU

        with torch.no_grad():
            # Tokenizer bucketize runs on CPU; keep context_tensor on CPU for tokenization
            context_tensor_cpu = context_tensor.cpu()
            tokenized = pipeline.tokenizer.context_input_transform(context_tensor_cpu)
            input_ids = tokenized[0].to(device)
            attention_mask = tokenized[1].to(device) if len(tokenized) > 1 else torch.ones_like(input_ids)

            encoder_outputs = encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            hidden = encoder_outputs.last_hidden_state  # (B, seq, hidden_dim)
            # Mean pool over sequence dimension (ignore padding via attention_mask)
            mask_expanded = attention_mask.unsqueeze(-1).float()
            pooled = (hidden * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1e-9)
            pooled_np = pooled.cpu().float().numpy()  # (B, hidden_dim)

        all_embeddings.append(pooled_np)

        if (start // chunk_size) % 20 == 0:
            print(f"    {min(start + chunk_size, total)}/{total} windows processed")
            sys.stdout.flush()

    return np.concatenate(all_embeddings, axis=0)  # (N, hidden_dim)


def main() -> None:
    args = build_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    env = collect_env()
    print(f"Device: {env['device_name']} | CUDA: {env['cuda_available']}")

    shared_start, shared_end = load_shared_window(args.metadata_path, args.no_shared_window)
    station_files = sorted(args.input_dir.glob("*_aligned_10min.csv"))
    if not station_files:
        raise FileNotFoundError(f"No station files in {args.input_dir}")

    print(f"Loading {len(station_files)} stations ...")
    frames, station_summaries, _, _ = build_all_stations(
        files=station_files,
        target=args.target,
        shared_start=shared_start,
        shared_end=shared_end,
        limit=args.limit_stations,
        max_steps=args.max_steps,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
    )

    context_steps = args.context_hours * steps_per_hour(args.freq)

    print(f"Loading Chronos pipeline from {args.teacher_model_path} ...")
    t0 = time.perf_counter()
    pipeline = load_chronos_pipeline(args.teacher_model_path, device)
    print(f"  Loaded in {time.perf_counter() - t0:.1f}s")

    # Fit PCA on train split embeddings, then apply to all splits
    pca: PCA | None = None
    results: dict[str, dict] = {}

    for split_name in ["train", "val", "test"]:
        index_path = args.teacher_cache_dir / f"window_index_{split_name}.csv"
        if not index_path.exists():
            print(f"  Skipping {split_name}: {index_path} not found")
            continue

        specs = load_window_specs(index_path)
        print(f"\nProcessing {split_name} ({len(specs)} windows) ...")
        t1 = time.perf_counter()

        embeddings = extract_encoder_embeddings(
            specs=specs,
            frames=frames,
            pipeline=pipeline,
            target=args.target,
            context_steps=context_steps,
            batch_size=args.batch_size,
            device=device,
        )
        print(f"  Embeddings shape: {embeddings.shape} | elapsed: {time.perf_counter()-t1:.1f}s")

        if split_name == "train":
            print(f"  Fitting PCA({args.n_components}) on train embeddings ...")
            pca = PCA(n_components=args.n_components, random_state=42)
            reduced = pca.fit_transform(embeddings)
            explained = pca.explained_variance_ratio_.sum()
            print(f"  Explained variance: {explained:.3f}")
        else:
            if pca is None:
                raise RuntimeError("PCA not fitted yet — train split must come first")
            reduced = pca.transform(embeddings)

        col_names = [f"enc_{i}" for i in range(args.n_components)]
        item_ids = [spec.item_id for spec in specs]
        out_df = pd.DataFrame(reduced, columns=col_names)
        out_df.insert(0, "item_id", item_ids)

        out_path = args.output_dir / f"encoder_features_{split_name}.csv"
        out_df.to_csv(out_path, index=False)
        print(f"  Saved {len(out_df)} rows to {out_path}")

        results[split_name] = {
            "windows": len(specs),
            "embedding_dim": int(embeddings.shape[1]),
            "pca_components": args.n_components,
            "explained_variance": float(pca.explained_variance_ratio_.sum()) if split_name == "train" else None,
            "output_csv": str(out_path),
            "elapsed_seconds": time.perf_counter() - t1,
        }

    log = {
        "created_at": str(pd.Timestamp.now()),
        "teacher_model": args.teacher_model_path,
        "context_hours": args.context_hours,
        "n_components": args.n_components,
        "splits": results,
    }
    log_path = args.output_dir / "encoder_features_log.json"
    log_path.write_text(json.dumps(log, indent=2))
    print(f"\nDone. Log: {log_path}")


if __name__ == "__main__":
    main()
