"""
train_distill_batch.py
======================
批量运行知识蒸馏实验：测试不同 α 权重（0.3, 0.5, 0.7）

这个脚本会：
1. 对每个 α 值训练一个蒸馏模型
2. 使用加权目标：target_distill = α * true + (1-α) * chronos_soft
3. 对比基线 PatchTST 和 Chronos 教师模型

用法：
    python train_distill_batch.py \
        --input-dir /root/autodl-tmp/processed_csv/aligned_stations \
        --soft-labels-dir /root/sota_runs/chronos_fullseq_labels \
        --output-dir /root/sota_runs/distill_runs \
        --horizon-hours 24 \
        --gpu-id 0
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=FutureWarning)
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def _read_cli_value(flag: str, default: str) -> str:
    try:
        idx = sys.argv.index(flag)
    except ValueError:
        return default
    if idx + 1 >= len(sys.argv):
        return default
    return sys.argv[idx + 1]


os.environ["CUDA_VISIBLE_DEVICES"] = _read_cli_value("--gpu-id", "0")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import numpy as np
import pandas as pd
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor
from autogluon.timeseries.metrics import check_get_evaluation_metric

try:
    import torch
except Exception:
    torch = None

DYNAMIC_COLUMNS = [
    "WDIR", "WSPD", "GST", "WVHT", "DPD", "APD", "MWD",
    "PRES", "ATMP", "WTMP", "DEWP", "VIS", "TIDE",
]
KNOWN_COVARIATES = [
    "time_sin_hour", "time_cos_hour",
    "time_sin_doy", "time_cos_doy",
    "month", "day_of_week",
]
AUXILIARY_COLUMNS = ["WVHT_true", "WVHT_distill", "chronos_pred"]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", type=Path, required=True)
    p.add_argument("--soft-labels-dir", type=Path, required=True)
    p.add_argument("--metadata-path", type=Path, default=None)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--horizon-hours", type=int, default=24)
    p.add_argument("--alphas", default="0.3,0.5,0.7",
                   help="蒸馏权重列表，逗号分隔")
    p.add_argument("--context-hours", type=int, default=168)
    p.add_argument("--freq", default="10min")
    p.add_argument("--train-ratio", type=float, default=0.70)
    p.add_argument("--val-ratio", type=float, default=0.20)
    p.add_argument("--time-limit", type=int, default=7200)
    p.add_argument("--gpu-id", default="0")
    p.add_argument("--limit-stations", type=int, default=0)
    p.add_argument("--speedup-factor", type=float, default=1.0,
                   help=">1 means faster training by reducing batches per epoch, e.g. 2.5")
    return p


def drop_auxiliary_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(columns=AUXILIARY_COLUMNS, errors="ignore")


def load_station(path: Path, shared_start, shared_end) -> pd.DataFrame:
    """加载单个站点数据"""
    item_id = path.stem.split("_")[0]
    cols_all = pd.read_csv(path, nrows=0).columns.tolist()
    dyn = [c for c in DYNAMIC_COLUMNS if c in cols_all]
    kno = [c for c in KNOWN_COVARIATES if c in cols_all]

    df = pd.read_csv(
        path,
        usecols=["datetime"] + dyn + kno,
        parse_dates=["datetime"],
        low_memory=False,
    )
    df = df.sort_values("datetime").drop_duplicates("datetime", keep="last").reset_index(drop=True)
    if shared_start is not None:
        df = df[(df["datetime"] >= shared_start) & (df["datetime"] <= shared_end)].reset_index(drop=True)

    num_cols = [c for c in df.columns if c != "datetime"]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("float32")
    df[num_cols] = df[num_cols].ffill().bfill()

    df.insert(0, "item_id", item_id)
    df = df.rename(columns={"datetime": "timestamp"})
    return df


def load_soft_label_cache(soft_labels_path: Path) -> dict[tuple[str, str], pd.DataFrame]:
    """一次性加载 Chronos 软标签，并按 (item_id, split) 建立缓存。"""
    if not soft_labels_path.exists():
        raise FileNotFoundError(f"软标签文件不存在：{soft_labels_path}")

    df = pd.read_csv(soft_labels_path, parse_dates=["timestamp"])
    df["item_id"] = df["item_id"].astype(str)
    df["split"] = df["split"].astype(str)

    cache: dict[tuple[str, str], pd.DataFrame] = {}
    for (item_id, split), sub in df.groupby(["item_id", "split"], sort=False):
        cache[(str(item_id), str(split))] = (
            sub[["timestamp", "chronos_pred"]]
            .sort_values("timestamp")
            .reset_index(drop=True)
        )
    return cache


def get_soft_labels(
    soft_label_cache: dict[tuple[str, str], pd.DataFrame],
    item_id: str,
    split: str,
) -> pd.DataFrame:
    """从缓存中读取 Chronos 软标签，缺少 val/test 时 fallback 到 train。"""
    item_id_str = str(item_id)

    sub = soft_label_cache.get((item_id_str, split))
    if sub is None:
        # fallback：用 train split
        sub = soft_label_cache.get((item_id_str, "train"))
        if sub is None:
            raise ValueError(f"软标签文件中找不到 item_id={item_id}")
        print(f"    [软标签] {item_id} split={split} 不存在，使用 train split 代替")

    return sub.copy()


def create_distill_target(df: pd.DataFrame, soft_df: pd.DataFrame, alpha: float) -> pd.DataFrame:
    """
    创建蒸馏目标：target_distill = α * true + (1-α) * soft
    """
    merged = df.merge(soft_df, on="timestamp", how="left")

    # 填充缺失的软标签（用真实值）
    merged["chronos_pred"] = merged["chronos_pred"].fillna(merged["WVHT"])

    # 保存原始 WVHT 用于评估
    merged["WVHT_true"] = merged["WVHT"].copy()

    # 用蒸馏目标替换 WVHT（训练目标）
    merged["WVHT"] = alpha * merged["WVHT"] + (1 - alpha) * merged["chronos_pred"]

    # 删除辅助列，避免 AutoGluon 把它们当特征
    merged = merged.drop(columns=["chronos_pred"], errors="ignore")

    return merged


def cleanup_alpha_dir(model_dir: Path) -> None:
    """Remove large AutoGluon artifacts after one alpha finishes."""
    if not model_dir.exists():
        return

    keep = {"result.json", "leaderboard_val.csv"}
    for child in model_dir.iterdir():
        if child.name in keep:
            continue
        if child.is_dir():
            shutil.rmtree(child, ignore_errors=True)
        else:
            child.unlink(missing_ok=True)


def load_existing_summary(summary_path: Path) -> dict[float, dict]:
    if not summary_path.exists():
        return {}
    try:
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception:
        return {}

    existing: dict[float, dict] = {}
    for item in payload.get("results", []):
        try:
            alpha = float(item.get("alpha"))
        except (TypeError, ValueError):
            continue
        existing[alpha] = item
    return existing


def preload_station_payloads(
    files: list[Path],
    soft_label_cache: dict[tuple[str, str], pd.DataFrame],
    shared_start,
    shared_end,
    train_ratio: float,
    val_ratio: float,
) -> list[dict]:
    """预加载站点数据与软标签，后续 alpha 循环直接复用内存缓存。"""
    station_payloads: list[dict] = []

    print("\n开始预加载站点数据和软标签缓存...")
    for path in files:
        item_id = path.stem.split("_")[0]
        print(f"\n[{item_id}] 加载数据...")

        df = load_station(path, shared_start, shared_end)
        n = len(df)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))

        try:
            soft_train = get_soft_labels(soft_label_cache, item_id, "train")
            soft_val = get_soft_labels(soft_label_cache, item_id, "val")
            soft_test = get_soft_labels(soft_label_cache, item_id, "test")
        except Exception as e:
            print(f"  跳过（软标签加载失败）：{e}")
            continue

        print(f"  Train: {train_end} | Val: {val_end - train_end} | Test: {n - val_end}")

        station_payloads.append(
            {
                "item_id": item_id,
                "df": df,
                "train_end": train_end,
                "val_end": val_end,
                "soft_train": soft_train,
                "soft_val": soft_val,
                "soft_test": soft_test,
            }
        )

    return station_payloads


def maybe_cleanup_cuda() -> None:
    gc.collect()
    if torch is None:
        return
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            if hasattr(torch.cuda, "ipc_collect"):
                torch.cuda.ipc_collect()
    except Exception:
        pass


def build_patchtst_hyperparameters(
    horizon_hours: int,
    context_length: int,
    speedup_factor: float,
) -> dict:
    """Build horizon-aware PatchTST hyperparameters with safer GPU usage for long horizons."""
    if horizon_hours >= 120:
        cfg = {
            "context_length": min(context_length, 720),
            "patch_len": 32,
            "stride": 16,
            "d_model": 64,
            "nhead": 4,
            "num_encoder_layers": 2,
            "dropout": 0.20,
            "lr": 1e-4,
            "batch_size": 2,
            "max_epochs": 120,
            "num_batches_per_epoch": 30,
            "early_stopping_patience": 15,
        }
    elif horizon_hours >= 72:
        cfg = {
            "context_length": min(context_length, 840),
            "patch_len": 32,
            "stride": 16,
            "d_model": 64,
            "nhead": 4,
            "num_encoder_layers": 2,
            "dropout": 0.20,
            "lr": 1e-4,
            "batch_size": 4,
            "max_epochs": 140,
            "num_batches_per_epoch": 40,
            "early_stopping_patience": 15,
        }
    elif horizon_hours >= 48:
        cfg = {
            "context_length": context_length,
            "patch_len": 16,
            "stride": 8,
            "d_model": 128,
            "nhead": 8,
            "num_encoder_layers": 4,
            "dropout": 0.15,
            "lr": 1e-4,
            "batch_size": 16,
            "max_epochs": 200,
            "num_batches_per_epoch": 100,
            "early_stopping_patience": 20,
        }
    elif horizon_hours >= 24:
        cfg = {
            "context_length": context_length,
            "patch_len": 16,
            "stride": 8,
            "d_model": 128,
            "nhead": 8,
            "num_encoder_layers": 4,
            "dropout": 0.15,
            "lr": 1e-4,
            "batch_size": 32,
            "max_epochs": 200,
            "num_batches_per_epoch": 150,
            "early_stopping_patience": 20,
        }
    else:
        cfg = {
            "context_length": context_length,
            "patch_len": 16,
            "stride": 8,
            "d_model": 128,
            "nhead": 8,
            "num_encoder_layers": 4,
            "dropout": 0.15,
            "lr": 1e-4,
            "batch_size": 128,
            "max_epochs": 200,
            "num_batches_per_epoch": 200,
            "early_stopping_patience": 20,
        }

    if speedup_factor > 1:
        cfg["num_batches_per_epoch"] = max(20, int(round(cfg["num_batches_per_epoch"] / speedup_factor)))
    return cfg


def save_result_and_cleanup(result: dict, model_dir: Path) -> None:
    result_path = model_dir / "result.json"
    result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    cleanup_alpha_dir(model_dir)


def train_one_alpha(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    test_raw_df: pd.DataFrame,
    alpha: float,
    horizon_hours: int,
    context_hours: int,
    freq: str,
    output_dir: Path,
    time_limit: int,
    speedup_factor: float,
) -> dict:
    """训练单个 α 值的蒸馏模型"""

    print(f"\n{'='*70}")
    print(f"训练蒸馏模型：α={alpha:.1f}, horizon={horizon_hours}h")
    print(f"  α={alpha:.1f} 表示：{alpha*100:.0f}% 真实标签 + {(1-alpha)*100:.0f}% Chronos 软标签")
    print(f"{'='*70}")

    steps_per_hour = {"10min": 6, "1h": 1, "30min": 2}.get(freq, 6)
    pred_length = horizon_hours * steps_per_hour
    ctx_length = context_hours * steps_per_hour

    # 转换为 TimeSeriesDataFrame，并移除蒸馏辅助列
    train_ts = TimeSeriesDataFrame.from_data_frame(
        drop_auxiliary_columns(train_df), id_column="item_id", timestamp_column="timestamp"
    )
    val_ts = TimeSeriesDataFrame.from_data_frame(
        drop_auxiliary_columns(val_df), id_column="item_id", timestamp_column="timestamp"
    )
    test_ts = TimeSeriesDataFrame.from_data_frame(
        drop_auxiliary_columns(test_df), id_column="item_id", timestamp_column="timestamp"
    )

    # 创建 predictor
    model_dir = output_dir / f"alpha_{alpha:.1f}"
    predictor = TimeSeriesPredictor(
        target="WVHT",
        prediction_length=pred_length,
        freq=freq,
        eval_metric="MASE",
        known_covariates_names=KNOWN_COVARIATES,
        path=str(model_dir),
    )

    # PatchTST 超参数（长时域使用更保守的显存配置）
    patchtst_cfg = build_patchtst_hyperparameters(
        horizon_hours=horizon_hours,
        context_length=ctx_length,
        speedup_factor=speedup_factor,
    )
    hyperparameters = {
        "PatchTST": patchtst_cfg
    }

    print(
        "  training_config: "
        f"batch_size={patchtst_cfg['batch_size']}, "
        f"context_length={patchtst_cfg['context_length']}, "
        f"d_model={patchtst_cfg['d_model']}, "
        f"nhead={patchtst_cfg['nhead']}, "
        f"num_encoder_layers={patchtst_cfg['num_encoder_layers']}, "
        f"num_batches_per_epoch={patchtst_cfg['num_batches_per_epoch']}, "
        f"speedup_factor={speedup_factor}"
    )

    print(f"\n开始训练（时间限制：{time_limit}秒）...")
    t0 = time.time()
    maybe_cleanup_cuda()

    try:
        predictor.fit(
            train_data=train_ts,
            tuning_data=val_ts,
            hyperparameters=hyperparameters,
            time_limit=time_limit,
            enable_ensemble=False,
        )
    except Exception as e:
        print(f"训练失败：{e}")
        import traceback; traceback.print_exc()
        raise  # 直接抛出，让外层停止

    elapsed = time.time() - t0
    print(f"训练完成，耗时：{elapsed:.1f}秒")
    maybe_cleanup_cuda()

    available_models = []
    try:
        available_models = list(predictor.model_names())
    except Exception:
        available_models = []

    if not available_models:
        error_msg = "Trainer has no fit models that can predict."
        print(f"训练未产出可用模型：{error_msg}")
        result = {
            "alpha": alpha,
            "horizon_hours": horizon_hours,
            "status": "train_failed",
            "metrics": {"error": error_msg},
            "elapsed_seconds": elapsed,
            "train_samples": len(train_df),
            "val_samples": len(val_df),
            "test_samples": len(test_df),
            "stations": sorted(train_df["item_id"].unique().tolist()),
        }
        save_result_and_cleanup(result, model_dir)
        return result

    # 评估（使用原始真实标签）
    print("\n评估模型（使用真实标签）...")

    # 去掉蒸馏辅助列，避免 AutoGluon 把它们当 past_covariate
    test_raw_clean = drop_auxiliary_columns(test_raw_df)

    test_ts_eval = TimeSeriesDataFrame.from_data_frame(
        test_raw_clean, id_column="item_id", timestamp_column="timestamp"
    )

    # 构造 known_covariates
    future_df = predictor.make_future_data_frame(test_ts_eval)
    future_reset = future_df.reset_index()
    kc_lookup = test_raw_df[["item_id", "timestamp"] + KNOWN_COVARIATES].copy()
    kc_merged = future_reset.merge(kc_lookup, on=["item_id", "timestamp"], how="left")
    for col in KNOWN_COVARIATES:
        if kc_merged[col].isna().any():
            kc_merged[col] = kc_merged[col].ffill().bfill()
    kc_ts_eval = TimeSeriesDataFrame.from_data_frame(
        kc_merged, id_column="item_id", timestamp_column="timestamp"
    )

    metrics = {}
    try:
        # 用 predict 手动计算指标，避免 leaderboard known_covariates 兼容问题
        preds = predictor.predict(test_ts_eval, known_covariates=kc_ts_eval)
        pred_mean = preds["mean"] if "mean" in preds.columns else preds.iloc[:, 0]

        # 对齐真实值和预测值
        all_true, all_pred = [], []
        for item_id in test_raw_df["item_id"].unique():
            item_true = test_raw_df[test_raw_df["item_id"] == item_id].sort_values("timestamp")
            true_wvht = item_true["WVHT"].values[-pred_length:]
            try:
                pred_vals = pred_mean.loc[item_id].values.flatten()
                min_len = min(len(true_wvht), len(pred_vals))
                all_true.append(true_wvht[:min_len])
                all_pred.append(pred_vals[:min_len])
            except Exception:
                pass

        if all_true:
            true_arr = np.concatenate(all_true)
            pred_arr = np.concatenate(all_pred)
            mae = float(np.mean(np.abs(pred_arr - true_arr)))
            rmse = float(np.sqrt(np.mean((pred_arr - true_arr) ** 2)))
            naive_mae = float(np.mean(np.abs(true_arr[1:] - true_arr[:-1])))
            mase = mae / (naive_mae + 1e-8)
            smape = float(np.mean(2 * np.abs(pred_arr - true_arr) / (np.abs(pred_arr) + np.abs(true_arr) + 1e-8))) * 100
            metrics = {"MAE": mae, "RMSE": rmse, "MASE": mase, "SMAPE": smape}

        # 也保存 leaderboard（val 上的）
        lb = predictor.leaderboard(display=False)
        lb_path = model_dir / "leaderboard_val.csv"
        lb.to_csv(lb_path, index=False)
        print(f"  Val leaderboard 已保存：{lb_path}")
        print(f"  测试集指标：{metrics}")
    except Exception as e:
        print(f"  评估失败：{e}")
        metrics = {"error": str(e)}

    status = "success" if "error" not in metrics else "eval_failed"

    result = {
        "alpha": alpha,
        "horizon_hours": horizon_hours,
        "status": status,
        "metrics": metrics,
        "elapsed_seconds": elapsed,
        "train_samples": len(train_df),
        "val_samples": len(val_df),
        "test_samples": len(test_df),
        "stations": sorted(train_df["item_id"].unique().tolist()),
    }

    save_result_and_cleanup(result, model_dir)
    maybe_cleanup_cuda()

    print(f"\n测试集指标：")
    for k, v in metrics.items():
        if v is not None:
            try:
                print(f"  {k}: {float(v):.4f}")
            except (TypeError, ValueError):
                print(f"  {k}: {v}")

    return result


def main():
    args = build_parser().parse_args()

    # 解析 alpha 列表
    alphas = [float(a.strip()) for a in args.alphas.split(",")]
    print(f"将测试 {len(alphas)} 个 α 值：{alphas}")

    # 加载 metadata
    if args.metadata_path and args.metadata_path.exists():
        meta = json.loads(args.metadata_path.read_text(encoding="utf-8"))
        shared_start = pd.Timestamp(meta["shared_start"])
        shared_end = pd.Timestamp(meta["shared_end"])
    else:
        shared_start = None
        shared_end = None

    # 软标签路径
    soft_labels_path = args.soft_labels_dir / f"labels_{args.horizon_hours:03d}h.csv"
    if not soft_labels_path.exists():
        raise FileNotFoundError(f"软标签文件不存在：{soft_labels_path}")

    print(f"加载软标签：{soft_labels_path}")
    soft_label_cache = load_soft_label_cache(soft_labels_path)

    # 加载所有站点数据
    files = sorted(args.input_dir.glob("*.csv"))
    if args.limit_stations > 0:
        files = files[:args.limit_stations]

    print(f"找到 {len(files)} 个站点文件")
    station_payloads = preload_station_payloads(
        files=files,
        soft_label_cache=soft_label_cache,
        shared_start=shared_start,
        shared_end=shared_end,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
    )
    if not station_payloads:
        raise RuntimeError("没有成功预加载任何站点数据，请检查输入数据和软标签。")

    # 为每个 alpha 准备数据
    summary_path = args.output_dir / f"horizon_{args.horizon_hours:03d}h" / "summary.json"
    existing_results = load_existing_summary(summary_path)
    all_results = []

    for alpha in alphas:
        if float(alpha) in existing_results:
            print(f"\n[SKIP] alpha={alpha:.1f} already exists")
            continue

        print(f"\n{'#'*70}")
        print(f"# 准备 α={alpha:.1f} 的数据")
        print(f"{'#'*70}")

        all_train = []
        all_val = []
        all_test = []
        all_test_raw = []

        for payload in station_payloads:
            item_id = payload["item_id"]
            df = payload["df"]
            train_end = payload["train_end"]
            val_end = payload["val_end"]
            soft_train = payload["soft_train"]
            soft_val = payload["soft_val"]
            soft_test = payload["soft_test"]

            print(f"\n[{item_id}] 复用预加载数据...")

            # 创建蒸馏目标
            df_train = create_distill_target(df.iloc[:train_end].copy(), soft_train, alpha)
            df_val = create_distill_target(df.iloc[train_end:val_end].copy(), soft_val, alpha)
            df_test = create_distill_target(df.iloc[val_end:].copy(), soft_test, alpha)

            # 保存原始 test（用真实 WVHT 评估）
            all_test_raw.append(df.iloc[val_end:].copy())

            print(f"  Train: {len(df_train)} | Val: {len(df_val)} | Test: {len(df_test)}")

            all_train.append(df_train)
            all_val.append(df_val)
            all_test.append(df_test)

        if not all_train:
            print(f"α={alpha:.1f} 没有可用数据，跳过")
            continue

        # 合并所有站点
        train_df = pd.concat(all_train, ignore_index=True)
        val_df = pd.concat(all_val, ignore_index=True)
        test_df = pd.concat(all_test, ignore_index=True)
        test_raw_df = pd.concat(all_test_raw, ignore_index=True)

        print(f"\n总数据量：")
        print(f"  Train: {len(train_df)} 行，{train_df['item_id'].nunique()} 个站点")
        print(f"  Val:   {len(val_df)} 行")
        print(f"  Test:  {len(test_df)} 行")

        # 训练模型
        output_dir = args.output_dir / f"horizon_{args.horizon_hours:03d}h"
        output_dir.mkdir(parents=True, exist_ok=True)

        result = train_one_alpha(
            train_df, val_df, test_df, test_raw_df,
            alpha=alpha,
            horizon_hours=args.horizon_hours,
            context_hours=args.context_hours,
            freq=args.freq,
            output_dir=output_dir,
            time_limit=args.time_limit,
            speedup_factor=args.speedup_factor,
        )

        all_results.append(result)
        existing_results[float(alpha)] = result

    # 保存汇总结果
    summary_path = args.output_dir / f"horizon_{args.horizon_hours:03d}h" / "summary.json"
    merged_results = [existing_results[k] for k in sorted(existing_results)]
    summary = {
        "horizon_hours": args.horizon_hours,
        "alphas_tested": sorted(existing_results),
        "results": merged_results,
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    # 打印对比表
    print(f"\n{'='*70}")
    print(f"蒸馏实验汇总 - {args.horizon_hours}h Horizon")
    print(f"{'='*70}")
    print(f"{'Alpha':<8} {'MASE':<10} {'MAE':<10} {'RMSE':<10} {'Status':<10}")
    print(f"{'-'*70}")

    for r in merged_results:
        if r["status"] == "success":
            mase = r["metrics"].get("MASE", 0)
            mae = r["metrics"].get("MAE", 0)
            rmse = r["metrics"].get("RMSE", 0)
            print(f"{r['alpha']:<8.1f} {mase:<10.4f} {mae:<10.4f} {rmse:<10.4f} {r['status']:<10}")
        else:
            print(f"{r['alpha']:<8.1f} {'N/A':<10} {'N/A':<10} {'N/A':<10} {r['status']:<10}")

    print(f"\n汇总结果已保存：{summary_path}")


if __name__ == "__main__":
    main()
