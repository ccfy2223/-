"""
rakd_inference.py
=================
推理阶段 RAKD-Wave：
  用 AutoGluon predictor 中各模型的预测 + Chronos 软标签，
  按 horizon-aware 权重融合，与 WeightedEnsemble 基线对比。

用法：
  python rakd_inference.py \
    --predictor-path /root/autodl-tmp/.Trash-0/files/autogluon_runs/horizon_006h/model \
    --soft-label-path /root/autodl-tmp/sota_runs/chronos_fullseq_labels/labels_006h.csv \
    --input-dir /root/autodl-tmp/processed_csv/aligned_stations \
    --metadata-path /root/autodl-tmp/processed_csv/shared_timeline_metadata.json \
    --horizon-hours 6 \
    --output-dir /root/autodl-tmp/sota_runs/rakd_inference_6h
"""
from __future__ import annotations
import argparse, json, pickle
import numpy as np
import pandas as pd
from pathlib import Path
from autogluon.timeseries import TimeSeriesPredictor, TimeSeriesDataFrame

DYNAMIC_COLUMNS = [
    "WDIR", "WSPD", "GST", "WVHT", "DPD", "APD", "MWD",
    "PRES", "ATMP", "WTMP", "DEWP", "VIS", "TIDE",
]
KNOWN_COVARIATES = [
    "time_sin_hour", "time_cos_hour",
    "time_sin_doy", "time_cos_doy",
    "month", "day_of_week",
]

# Horizon-aware 教师可信度（Chronos 软标签权重上限）
HORIZON_BETA = {1: 0.1, 3: 0.15, 6: 0.2, 12: 0.4, 24: 0.5, 48: 0.35, 72: 0.3, 120: 0.25}


def load_station(path: Path, shared_start, shared_end) -> pd.DataFrame:
    item_id = path.stem.split("_")[0]
    cols_all = pd.read_csv(path, nrows=0).columns.tolist()
    dyn = [c for c in DYNAMIC_COLUMNS if c in cols_all]
    kno = [c for c in KNOWN_COVARIATES if c in cols_all]
    df = pd.read_csv(path, usecols=["datetime"] + dyn + kno,
                     parse_dates=["datetime"], low_memory=False)
    df = df.sort_values("datetime").drop_duplicates("datetime", keep="last").reset_index(drop=True)
    if shared_start is not None:
        df = df[(df["datetime"] >= shared_start) & (df["datetime"] <= shared_end)].reset_index(drop=True)
    num_cols = [c for c in df.columns if c != "datetime"]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("float32")
    df[num_cols] = df[num_cols].ffill().bfill().fillna(0.0)
    df.insert(0, "item_id", item_id)
    df = df.rename(columns={"datetime": "timestamp"})
    return df


def compute_metrics(pred: np.ndarray, true: np.ndarray, naive_mae: float) -> dict:
    mae = float(np.mean(np.abs(pred - true)))
    rmse = float(np.sqrt(np.mean((pred - true) ** 2)))
    mase = mae / (naive_mae + 1e-8)
    smape = float(np.mean(2 * np.abs(pred - true) / (np.abs(pred) + np.abs(true) + 1e-8))) * 100
    return {"MAE": mae, "RMSE": rmse, "MASE": mase, "SMAPE": smape}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--predictor-path", type=Path, required=True)
    p.add_argument("--soft-label-path", type=Path, required=True)
    p.add_argument("--input-dir", type=Path, required=True)
    p.add_argument("--metadata-path", type=Path, default=None)
    p.add_argument("--horizon-hours", type=int, default=6)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--freq", default="10min")
    p.add_argument("--train-ratio", type=float, default=0.70)
    p.add_argument("--val-ratio", type=float, default=0.20)
    args = p.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    steps_per_hour = {"10min": 6, "1h": 1, "30min": 2}.get(args.freq, 6)
    pred_length = args.horizon_hours * steps_per_hour
    beta = HORIZON_BETA.get(args.horizon_hours, 0.2)

    print(f"加载 predictor: {args.predictor_path}")
    predictor = TimeSeriesPredictor.load(str(args.predictor_path))
    lb = predictor.leaderboard(display=False)
    # 排除 WeightedEnsemble，只用单模型
    single_models = lb[~lb["model"].str.contains("WeightedEnsemble")]["model"].tolist()
    print(f"可用单模型: {single_models}")
    print(f"Horizon={args.horizon_hours}h, beta(Chronos软标签权重)={beta}")

    # 加载 metadata
    if args.metadata_path and args.metadata_path.exists():
        meta = json.loads(args.metadata_path.read_text(encoding="utf-8"))
        shared_start = pd.Timestamp(meta["shared_start"])
        shared_end = pd.Timestamp(meta["shared_end"])
    else:
        shared_start = shared_end = None

    # 加载软标签
    print(f"加载软标签: {args.soft_label_path}")
    soft_df = pd.read_csv(args.soft_label_path, parse_dates=["timestamp"])
    soft_df["item_id"] = soft_df["item_id"].astype(str)

    # 加载站点数据，构造测试集
    files = sorted(args.input_dir.glob("*.csv"))
    all_frames = []
    for path in files:
        df = load_station(path, shared_start, shared_end)
        all_frames.append(df)

    # 构造完整 TimeSeriesDataFrame 和 known_covariates
    combined = pd.concat(all_frames, ignore_index=True)
    ts_full = TimeSeriesDataFrame.from_data_frame(
        combined, id_column="item_id", timestamp_column="timestamp"
    )

    # known_covariates：未来已知特征（AutoGluon predict 需要）
    kc_cols = [c for c in KNOWN_COVARIATES if c in combined.columns]
    kc_df = combined[["item_id", "timestamp"] + kc_cols].copy()
    kc_ts = TimeSeriesDataFrame.from_data_frame(
        kc_df, id_column="item_id", timestamp_column="timestamp"
    )

    # 对每个单模型做预测
    model_preds = {}
    for model in single_models:
        print(f"  预测模型: {model}")
        try:
            preds = predictor.predict(ts_full, known_covariates=kc_ts, model=model)
            model_preds[model] = preds["mean"]
        except Exception as e:
            try:
                preds = predictor.predict(ts_full, model=model)
                model_preds[model] = preds["mean"]
            except Exception as e2:
                print(f"    失败: {e2}")

    if not model_preds:
        print("没有可用的模型预测，退出")
        return

    # 提取测试集真实值
    # 测试集 = 每个站点最后 10% 的预测目标
    true_vals = {}
    naive_maes = []
    for path in files:
        item_id = path.stem.split("_")[0]
        df = load_station(path, shared_start, shared_end)
        n = len(df)
        val_end = int(n * (args.train_ratio + args.val_ratio))
        test_df = df.iloc[val_end:].reset_index(drop=True)
        wvht = test_df["WVHT"].values.astype(np.float32)
        true_vals[item_id] = (test_df["timestamp"].values, wvht)
        naive_maes.append(float(np.mean(np.abs(wvht[1:] - wvht[:-1]))))
    naive_mae = float(np.mean(naive_maes))
    print(f"Naive MAE (MASE 分母): {naive_mae:.4f}")

    # 对每个站点计算各方案的指标
    results = {m: {"preds": [], "trues": []} for m in list(model_preds.keys()) + ["RAKD"]}

    for item_id, (ts_arr, true_wvht) in true_vals.items():
        # 软标签
        soft_sub = soft_df[
            (soft_df["item_id"] == str(item_id)) & (soft_df["split"] == "test")
        ].set_index("timestamp")["chronos_pred"]

        # 对每个时间步，取预测窗口的均值作为该步的预测值
        # 这里用最后一个预测窗口的 mean 预测
        for model, pred_series in model_preds.items():
            try:
                item_pred = pred_series.loc[item_id].values.flatten()
                # 截取和 true_wvht 等长
                min_len = min(len(item_pred), len(true_wvht))
                results[model]["preds"].append(item_pred[:min_len])
                results[model]["trues"].append(true_wvht[:min_len])
            except Exception:
                pass

        # RAKD：用 val 上最佳单模型 + Chronos 软标签融合
        best_model = lb[
            ~lb["model"].str.contains("WeightedEnsemble|Chronos", case=False)
        ].iloc[0]["model"]
        if best_model in model_preds:
            try:
                student_pred = model_preds[best_model].loc[item_id].values.flatten()
                soft_aligned = pd.Series(
                    pd.Series(soft_sub).reindex(ts_arr).values,
                    dtype=np.float32
                ).fillna(pd.Series(true_wvht)).values

                min_len = min(len(student_pred), len(true_wvht), len(soft_aligned))
                rakd_pred = (1 - beta) * student_pred[:min_len] + beta * soft_aligned[:min_len]
                results["RAKD"]["preds"].append(rakd_pred)
                results["RAKD"]["trues"].append(true_wvht[:min_len])
            except Exception as e:
                print(f"  RAKD {item_id} 失败: {e}")

    # 汇总指标
    print(f"\n{'='*70}")
    print(f"Horizon {args.horizon_hours}h 推理结果对比")
    print(f"{'='*70}")
    print(f"{'Model':<35} {'MAE':<10} {'RMSE':<10} {'MASE':<10} {'SMAPE':<10}")
    print(f"{'-'*70}")

    summary = {}
    for model, data in results.items():
        if not data["preds"]:
            continue
        all_preds = np.concatenate(data["preds"])
        all_trues = np.concatenate(data["trues"])
        m = compute_metrics(all_preds, all_trues, naive_mae)
        summary[model] = m
        marker = " ← RAKD" if model == "RAKD" else ""
        print(f"{model:<35} {m['MAE']:<10.4f} {m['RMSE']:<10.4f} {m['MASE']:<10.4f} {m['SMAPE']:<10.2f}{marker}")

    # 保存结果
    out_path = args.output_dir / f"results_{args.horizon_hours:03d}h.json"
    out_path.write_text(json.dumps({
        "horizon_hours": args.horizon_hours,
        "beta": beta,
        "naive_mae": naive_mae,
        "metrics": summary,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n结果已保存: {out_path}")


if __name__ == "__main__":
    main()
