# wave_predict_core

Core code for multi-station significant wave height forecasting and knowledge distillation.

This repository is a cleaned code subset extracted from the original thesis project. It keeps the main training, distillation, preprocessing, and plotting scripts, while excluding large datasets, model checkpoints, logs, archives, and thesis build artifacts.

## Overview

The project focuses on significant wave height (SWH) forecasting using long-term NDBC buoy observations from multiple sea regions. The main workflow contains three parts:

1. Data preprocessing  
   Download raw buoy data, align all stations to a shared 10-minute timeline, and generate unified station-level or merged datasets.

2. Baseline forecasting  
   Train and compare multiple time-series models, including PatchTST, Temporal Fusion Transformer (TFT), DLinear, DeepAR, and Chronos, under the same train/validation/test split.

3. Knowledge distillation  
   Use Chronos as the teacher model and a lightweight student model such as PatchTST or TFT to test whether soft labels can improve forecasting performance across different horizons.

## What is included

- Main training scripts
- Distillation and soft-label generation scripts
- Preprocessing scripts
- Plotting and summary scripts
- Helper scripts used for thesis figures and result aggregation

## What is not included

- Large processed datasets
- Model weights and checkpoints
- Training outputs and experiment folders
- Logs, compressed archives, and temporary files
- Thesis PDF and LaTeX build artifacts

## Repository structure

```text
wave_predict_core/
├─ train_autogluon.py
├─ train_distill.py
├─ train_distill_batch.py
├─ train_distill_pytorch.py
├─ train_tft_chronos2_cascade.py
├─ train_tft_distill.py
├─ train_tft_iterative.py
├─ train_unified.py
├─ generate_chronos_teacher.py
├─ generate_chronos_fullseq_labels.py
├─ generate_chronos_encoder_features.py
├─ run_distill_after_labels.py
├─ run_softlabels_3gpu.py
├─ run_softlabels_then_distill_3gpu.py
├─ run_softlabels_then_distill_4gpu_short.py
├─ prepare_timebridge_wave_dataset.py
├─ run_timebridge_wave.py
├─ run_timeemb_wave.py
├─ plot_results.py
├─ plot_predictions.py
├─ plot_training_curves.py
├─ generate_paper_figures.py
├─ regen_plots.py
├─ make_summary_table.py
├─ hpo_distill.py
├─ rakd_inference.py
├─ local_pretrained_cascade_infer.py
├─ plot_local_cascade_forecasts.py
├─ check_torch_env.py
├─ scripts/
├─ data_prep/
├─ CHRONOS_TEACHER_DEBUG_GUIDE.md
└─ README.md
```

## Main scripts

### 1. Baseline training

- `train_autogluon.py`  
  Main baseline script for multi-horizon forecasting with AutoGluon TimeSeries.

- `train_unified.py`  
  Unified training entry for related experiments.

### 2. Distillation

- `train_distill.py`  
  Main distillation training script.

- `train_distill_batch.py`  
  Batch experiment runner for different distillation weights.

- `train_distill_pytorch.py`  
  PyTorch-based distillation implementation.

- `train_tft_distill.py`  
  Distillation experiments built around TFT.

- `run_distill_after_labels.py`  
  Waits for teacher labels to be ready and then launches distillation.

### 3. Teacher-label generation

- `generate_chronos_teacher.py`
- `generate_chronos_fullseq_labels.py`
- `generate_chronos_encoder_features.py`

These scripts generate Chronos-based teacher outputs or teacher-side features used in downstream experiments.

### 4. Iterative and cascade forecasting

- `train_tft_chronos2_cascade.py`
- `train_tft_iterative.py`
- `local_pretrained_cascade_infer.py`
- `rakd_inference.py`

These scripts are used for iterative prediction, cascade forecasting, and local inference experiments.

### 5. Data preprocessing

Files in `data_prep/` are copied from the original preprocessing folder:

- `download_8_typical_ndbc.py`
- `inspect_inputs.py`
- `align_ndbc_timelines.py`
- `build_station_overview.py`
- `export_processed_csv.py`

They are used to download, inspect, align, and export the multi-station buoy dataset.

### 6. Plotting and result summarization

- `plot_results.py`
- `plot_predictions.py`
- `plot_training_curves.py`
- `generate_paper_figures.py`
- `regen_plots.py`
- `make_summary_table.py`
- scripts under `scripts/`

These scripts are mainly used to generate thesis figures, model comparison plots, and summary tables.

## Recommended workflow

### Step 1: prepare data

Use the scripts in `data_prep/` to download and align the NDBC station data.

### Step 2: train baseline models

Run:

```bash
python train_autogluon.py
```

Adjust input path, output path, GPU id, and horizons as needed.

### Step 3: generate teacher labels

Run the Chronos-related generation scripts, for example:

```bash
python generate_chronos_fullseq_labels.py
```

### Step 4: run distillation

For example:

```bash
python train_distill_batch.py
```

### Step 5: generate figures and summaries

Run plotting and summary scripts after training is complete.

## Environment notes

This project was originally developed in a Windows-centered workflow, with some training runs moved to AutoDL Linux.

Common dependencies include:

- Python 3.10 or 3.11
- pandas
- numpy
- matplotlib
- torch
- autogluon.timeseries
- lightning
- optuna
- plotly
- scikit-learn
- pillow

Because the original project evolved over multiple experiments, not every script uses the exact same dependency set. It is recommended to install dependencies incrementally based on the scripts you plan to run.

## Notes for GitHub use

This repository is intended to keep only the core code. If you want full reproduction, you still need to prepare:

- the aligned buoy dataset
- Chronos teacher outputs or soft labels
- experiment configuration paths
- model checkpoints when required

If you upload this repository to GitHub, keep `.gitignore` enabled so that large outputs are not pushed by mistake.

## Original source

This code subset was copied from the original local thesis project for easier versioning and GitHub upload.
