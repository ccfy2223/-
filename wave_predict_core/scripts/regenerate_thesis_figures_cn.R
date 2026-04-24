#!/usr/bin/env Rscript

options(stringsAsFactors = FALSE)

local_lib <- normalizePath(file.path(getwd(), ".r_libs"), winslash = "/", mustWork = FALSE)
if (dir.exists(local_lib)) {
  .libPaths(c(local_lib, .libPaths()))
}

suppressPackageStartupMessages({
  library(ggplot2)
  library(dplyr)
  library(cowplot)
  library(scales)
  library(maps)
  library(grid)
})

cmd_args <- commandArgs(trailingOnly = FALSE)
script_arg <- cmd_args[grep("^--file=", cmd_args)][1]
script_path <- normalizePath(sub("^--file=", "", script_arg), winslash = "/", mustWork = TRUE)
root_dir <- normalizePath(file.path(dirname(script_path), ".."), winslash = "/", mustWork = TRUE)
fig_dir <- file.path(root_dir, "thesis_assets", "figures")
table_dir <- file.path(root_dir, "thesis_assets", "tables")
latex_fig_dir <- file.path(root_dir, "-2026-LaTeX--main", "figures", "from_111md")
curve_root <- file.path(root_dir, "autogluon_runs_local_3060")

dir.create(fig_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(latex_fig_dir, recursive = TRUE, showWarnings = FALSE)

theme_set(
  theme_minimal(base_family = "Microsoft YaHei", base_size = 12) +
    theme(
      plot.title = element_text(face = "bold", size = 15, colour = "#17324D"),
      plot.subtitle = element_text(size = 10.5, colour = "#516B84"),
      axis.title = element_text(face = "bold", colour = "#25435E"),
      axis.text = element_text(colour = "#35556F"),
      legend.title = element_text(face = "bold"),
      legend.position = "bottom",
      panel.grid.minor = element_blank(),
      panel.grid.major = element_line(colour = "#DCE6EE", linewidth = 0.35),
      strip.text = element_text(face = "bold", colour = "#17324D"),
      plot.background = element_rect(fill = "white", colour = NA),
      panel.background = element_rect(fill = "white", colour = NA)
    )
)

model_levels <- c("Chronos", "TFT", "PatchTST", "DLinear", "DeepAR", "加权集成")
model_palette <- c(
  "Chronos" = "#0E5A8A",
  "TFT" = "#D97A2B",
  "PatchTST" = "#2E8B57",
  "DLinear" = "#8A6BBE",
  "DeepAR" = "#B64E59",
  "加权集成" = "#5E6C7A"
)

nice_model_name <- function(x) {
  x <- gsub("^Chronos\\[amazon__chronos-t5-large\\]$", "Chronos", x)
  x <- gsub("^TemporalFusionTransformer$", "TFT", x)
  x <- gsub("^WeightedEnsemble$", "加权集成", x)
  x
}

save_dual <- function(plot_obj, filename, width, height, dpi = 320, bg = "white") {
  out1 <- file.path(fig_dir, filename)
  out2 <- file.path(latex_fig_dir, filename)
  ggsave(out1, plot_obj, width = width, height = height, dpi = dpi, bg = bg)
  file.copy(out1, out2, overwrite = TRUE)
}

metrics_summary <- read.csv(file.path(table_dir, "export_metrics_summary.csv"), check.names = FALSE)
metrics_summary$best_model <- nice_model_name(metrics_summary$best_model)

leaderboard_files <- list.files(table_dir, pattern = "^leaderboard_[0-9]{3}h\\.csv$", full.names = TRUE)
leaderboards <- bind_rows(lapply(leaderboard_files, function(fp) {
  h <- as.integer(sub(".*leaderboard_([0-9]{3})h\\.csv$", "\\1", fp))
  df <- read.csv(fp, check.names = FALSE)
  df$horizon_hours <- h
  df$model <- nice_model_name(df$model)
  df
}))

leaderboards <- leaderboards %>%
  filter(model %in% c("Chronos", "TFT", "PatchTST", "DLinear", "DeepAR", "加权集成")) %>%
  mutate(
    model = factor(model, levels = model_levels),
    horizon_label = factor(paste0(horizon_hours, "h"), levels = paste0(sort(unique(horizon_hours)), "h"))
  )

station_split <- read.csv(file.path(table_dir, "station_split_summary.csv"), check.names = FALSE)
selected_stations <- read.csv(file.path(table_dir, "selected_stations.csv"), check.names = FALSE)
selected_stations_ref <- selected_stations[, c("station_id", "station_name", "profile")]
names(selected_stations_ref)[names(selected_stations_ref) == "station_name"] <- "station_name_ref"

station_plot_df <- station_split %>%
  left_join(selected_stations_ref, by = c("item_id" = "station_id")) %>%
  mutate(
    label = as.character(item_id),
    station_name = ifelse(is.na(station_name_ref) | station_name_ref == "", item_id, station_name_ref),
    train_rows = as.numeric(train_rows),
    val_rows = as.numeric(val_rows),
    test_rows = as.numeric(test_rows)
  )

split_long <- bind_rows(
  transform(station_plot_df[, c("item_id", "station_name", "train_rows")], split = "训练集", rows = train_rows),
  transform(station_plot_df[, c("item_id", "station_name", "val_rows")], split = "验证集", rows = val_rows),
  transform(station_plot_df[, c("item_id", "station_name", "test_rows")], split = "测试集", rows = test_rows)
)

split_long$label <- paste0(split_long$item_id)

p_split <- ggplot(split_long, aes(x = label, y = rows / 1000, fill = split)) +
  geom_col(width = 0.72, colour = "white", linewidth = 0.2) +
  scale_fill_manual(values = c("训练集" = "#0E5A8A", "验证集" = "#68A6C8", "测试集" = "#D97A2B")) +
  labs(
    title = "A. 多站点样本划分",
    subtitle = "统一共享时间窗内，各站点样本量一致",
    x = "浮标站点",
    y = "样本数（千条）",
    fill = "数据划分"
  ) +
  geom_text(
    data = station_plot_df,
    aes(x = label, y = (train_rows + val_rows + test_rows) / 1000 + 20, label = paste0("总计 ", round((train_rows + val_rows + test_rows) / 1000), "k")),
    inherit.aes = FALSE,
    size = 3.3,
    colour = "#17324D"
  ) +
  theme(
    axis.text.x = element_text(angle = 0, hjust = 0.5),
    legend.position = "top"
  )

missing_long <- bind_rows(
  transform(station_plot_df[, c("item_id", "missing_before_fill")], stage = "填充前", missing = missing_before_fill),
  transform(station_plot_df[, c("item_id", "missing_after_fill")], stage = "填充后", missing = missing_after_fill)
)
missing_long$label <- paste0(missing_long$item_id)

p_missing <- ggplot(missing_long, aes(x = label, y = missing / 1e6, fill = stage)) +
  geom_col(position = position_dodge(width = 0.72), width = 0.66, colour = "white", linewidth = 0.2) +
  scale_fill_manual(values = c("填充前" = "#B64E59", "填充后" = "#68A6C8")) +
  labs(
    title = "B. 缺失值处理前后对比",
    subtitle = "残余缺失主要来自 VIS、TIDE 等后续剔除协变量",
    x = "浮标站点",
    y = "缺失项数量（百万）",
    fill = "处理阶段"
  ) +
  theme(legend.position = "top")

p1 <- plot_grid(p_split, p_missing, ncol = 1, rel_heights = c(1, 1.02), align = "v")
title_p1 <- ggdraw() +
  draw_label("图2-2 多站点数据划分与缺失值处理概览", x = 0.02, hjust = 0, fontface = "bold", size = 16, colour = "#17324D") +
  draw_label("样本划分、共享时间窗与缺失值清洗过程的可视化汇总", x = 0.02, y = 0.28, hjust = 0, size = 10.5, colour = "#516B84")
p1 <- plot_grid(title_p1, p1, ncol = 1, rel_heights = c(0.12, 1))
save_dual(p1, "fig1_dataset_overview.png", width = 11.5, height = 8.2)

plot_metric_line <- function(metric_col, y_label, filename, title_text) {
  sub <- leaderboards %>%
    filter(!is.na(.data[[metric_col]]), model %in% c("Chronos", "TFT", "PatchTST", "DLinear", "DeepAR")) %>%
    arrange(horizon_hours)

  g <- ggplot(sub, aes(x = horizon_hours, y = .data[[metric_col]], colour = model)) +
    geom_line(linewidth = 1.1) +
    geom_point(size = 2.6) +
    scale_colour_manual(values = model_palette[levels(droplevels(sub$model))]) +
    scale_x_continuous(breaks = sort(unique(sub$horizon_hours))) +
    labs(
      title = title_text,
      subtitle = "五类基线模型在统一测试集上的误差随预测时域变化趋势",
      x = "预测时域（小时）",
      y = y_label,
      colour = "模型"
    )
  save_dual(g, filename, width = 9.8, height = 5.8)
}

plot_metric_line("MASE", "MASE", "fig2_comparison_mase.png", "图3-2 不同模型 MASE 随预测时域变化趋势")
plot_metric_line("MAE", "MAE（米）", "fig2_comparison_mae.png", "不同模型 MAE 随预测时域变化趋势")
plot_metric_line("RMSE", "RMSE（米）", "fig2_comparison_rmse.png", "不同模型 RMSE 随预测时域变化趋势")
plot_metric_line("SMAPE", "SMAPE", "fig2_comparison_smape.png", "不同模型 SMAPE 随预测时域变化趋势")

heat_df <- leaderboards %>%
  filter(model %in% c("Chronos", "TFT", "PatchTST", "DLinear", "DeepAR")) %>%
  select(model, horizon_label, MASE)

p_heat <- ggplot(heat_df, aes(x = horizon_label, y = model, fill = MASE)) +
  geom_tile(colour = "white", linewidth = 0.6) +
  geom_text(aes(label = sprintf("%.3f", MASE)), size = 3.2, colour = "#17324D") +
  scale_fill_gradient(low = "#E8F3F8", high = "#0E5A8A") +
  labs(
    title = "图3-5 各模型 MASE 热力图",
    subtitle = "颜色越深表示误差越大，数值标注为测试集 MASE",
    x = "预测时域",
    y = "模型",
    fill = "MASE"
  )
save_dual(p_heat, "fig4_mase_heatmap.png", width = 9.6, height = 5.6)

val_test_df <- leaderboards %>%
  filter(model %in% c("Chronos", "TFT", "PatchTST", "DLinear", "DeepAR")) %>%
  select(model, horizon_hours, validation_MASE, MASE) %>%
  rename(`验证集` = validation_MASE, `测试集` = MASE)

val_test_long <- bind_rows(
  transform(val_test_df[, c("model", "horizon_hours", "验证集")], split = "验证集", value = 验证集),
  transform(val_test_df[, c("model", "horizon_hours", "测试集")], split = "测试集", value = 测试集)
) %>%
  mutate(horizon_label = factor(paste0(horizon_hours, "h"), levels = paste0(sort(unique(horizon_hours)), "h")))

p_vt <- ggplot(val_test_long, aes(x = horizon_label, y = value, fill = split)) +
  geom_col(position = position_dodge(width = 0.75), width = 0.68, colour = "white", linewidth = 0.2) +
  facet_wrap(~model, ncol = 3, scales = "free_y") +
  scale_fill_manual(values = c("验证集" = "#6BAED6", "测试集" = "#D97A2B")) +
  labs(
    title = "图3-6 验证集与测试集 MASE 对比",
    subtitle = "用于判断不同模型在多预测时域下的验证-测试一致性",
    x = "预测时域",
    y = "MASE",
    fill = "评估集合"
  )
save_dual(p_vt, "fig5_val_vs_test_mase.png", width = 10.8, height = 7.2)

p_time <- ggplot(metrics_summary, aes(x = factor(horizon_hours), y = elapsed_seconds / 60, fill = best_model)) +
  geom_col(width = 0.72, colour = "white", linewidth = 0.25) +
  geom_text(aes(label = sprintf("%.1f", elapsed_seconds / 60)), vjust = -0.25, size = 3.2, colour = "#17324D") +
  scale_fill_manual(values = model_palette[names(model_palette) %in% unique(metrics_summary$best_model)]) +
  labs(
    title = "图3-4 各预测时域训练时间对比",
    subtitle = "柱高表示该时域最终选择模型的总训练耗时",
    x = "预测时域（小时）",
    y = "训练时间（分钟）",
    fill = "最优模型"
  ) +
  theme(legend.position = "top")
save_dual(p_time, "fig7_training_time.png", width = 9.8, height = 5.8)

read_curve <- function(h, model) {
  fp <- file.path(curve_root, sprintf("horizon_%03dh", h), "training_curves", model, "version_0", "metrics.csv")
  if (!file.exists(fp)) {
    return(NULL)
  }
  df <- read.csv(fp, check.names = FALSE)
  df$model <- nice_model_name(model)
  df$horizon_hours <- h
  df$step_index <- seq_len(nrow(df))
  df
}

curve_models <- c("TemporalFusionTransformer", "PatchTST", "DLinear", "DeepAR")
curve_hours <- c(1, 3, 6, 12, 24, 48, 72, 120)
curve_df <- bind_rows(lapply(curve_hours, function(h) bind_rows(lapply(curve_models, function(m) read_curve(h, m)))))

curve_val <- curve_df %>%
  filter(!is.na(val_loss)) %>%
  group_by(model, horizon_hours) %>%
  arrange(step_index, .by_group = TRUE) %>%
  mutate(val_smooth = stats::filter(val_loss, rep(1 / 5, 5), sides = 1)) %>%
  ungroup()

rep_df <- curve_val %>% filter(horizon_hours %in% c(1, 24, 120))
p_rep <- ggplot(rep_df, aes(x = epoch, y = val_smooth, colour = model)) +
  geom_line(linewidth = 1.0, na.rm = TRUE) +
  facet_wrap(~horizon_hours, nrow = 1, scales = "free_y", labeller = function(x) paste0(x, "h")) +
  scale_colour_manual(values = model_palette[names(model_palette) %in% unique(rep_df$model)]) +
  labs(
    title = "图3-7 代表性预测时域验证损失收敛曲线",
    subtitle = "展示 1h、24h 与 120h 三个代表性时域下不同模型的训练收敛特征",
    x = "训练轮次（Epoch）",
    y = "平滑后的验证损失",
    colour = "模型"
  )
save_dual(p_rep, "training_curves_representative.png", width = 11.5, height = 4.6)

epochs_df <- curve_val %>%
  group_by(model, horizon_hours) %>%
  summarise(train_epochs = max(epoch, na.rm = TRUE) + 1, .groups = "drop")

p_epoch <- ggplot(epochs_df, aes(x = factor(horizon_hours), y = train_epochs, fill = model)) +
  geom_col(position = position_dodge(width = 0.78), width = 0.7, colour = "white", linewidth = 0.2) +
  scale_fill_manual(values = model_palette[names(model_palette) %in% unique(epochs_df$model)]) +
  labs(
    title = "图3-8 各模型实际训练轮数统计",
    subtitle = "以训练日志中记录的最大 epoch 估计 Early Stopping 后的真实训练轮数",
    x = "预测时域（小时）",
    y = "训练轮数",
    fill = "模型"
  )
save_dual(p_epoch, "training_epochs_early_stopping.png", width = 10.5, height = 5.8)

overview_script <- file.path(root_dir, "scripts", "rebuild_training_curves_overview.py")
if (file.exists(overview_script)) {
  system2(
    command = file.path(root_dir, ".venv311", "Scripts", "python.exe"),
    args = shQuote(overview_script),
    stdout = TRUE,
    stderr = TRUE
  )
}

forecast_path <- file.path(table_dir, "baseline_024h_prediction_compare.csv")
if (file.exists(forecast_path)) {
  forecast_df <- read.csv(forecast_path, check.names = FALSE)
  forecast_df$model <- factor(forecast_df$model, levels = c("Chronos", "TFT", "PatchTST", "DLinear", "DeepAR"))
  forecast_df$timestamp <- as.POSIXct(forecast_df$timestamp)

  pred_metrics <- forecast_df %>%
    group_by(model) %>%
    summarise(
      mae = mean(abs(y_pred - y_true), na.rm = TRUE),
      rmse = sqrt(mean((y_pred - y_true)^2, na.rm = TRUE)),
      y_pos = max(pmax(y_true, y_pred), na.rm = TRUE),
      x_pos = min(timestamp, na.rm = TRUE) + 0.985 * (max(timestamp, na.rm = TRUE) - min(timestamp, na.rm = TRUE)),
      .groups = "drop"
    ) %>%
    mutate(
      label = sprintf("MAE = %.3f\nRMSE = %.3f", mae, rmse)
    )

  p_pred <- ggplot(forecast_df, aes(x = timestamp)) +
    geom_line(aes(y = y_true), linewidth = 1.05, colour = "#111111") +
    geom_line(aes(y = y_pred, colour = model), linewidth = 0.95, alpha = 0.96) +
    geom_text(
      data = pred_metrics,
      aes(x = x_pos, y = y_pos, label = label),
      inherit.aes = FALSE,
      hjust = 1,
      vjust = 1.2,
      size = 3.25,
      colour = "#2D4052",
      lineheight = 0.95,
      family = "Times New Roman"
    ) +
    facet_wrap(~model, ncol = 2) +
    scale_colour_manual(values = model_palette[names(model_palette) %in% unique(forecast_df$model)]) +
    scale_x_datetime(date_labels = "%m-%d\n%H:%M", date_breaks = "6 hours") +
    labs(
      title = "24小时预测时域真实值与模型预测值对比",
      subtitle = "站点41010，同一测试窗口；黑线为真实值，彩色线为对应模型预测值",
      x = "时间",
      y = "有效波高（米）",
      colour = "模型"
    ) +
    theme(
      legend.position = "none",
      strip.text = element_text(face = "bold", size = 11.5),
      panel.spacing = unit(0.65, "lines"),
      panel.border = element_rect(colour = "#BFD0DE", fill = NA, linewidth = 0.45),
      axis.text.x = element_text(size = 9.2),
      plot.title = element_text(size = 16),
      plot.subtitle = element_text(size = 10.8)
    )
  save_dual(p_pred, "fig3_prediction_vs_truth_24h.png", width = 11.2, height = 8.6)

  eqn_annot <- forecast_df %>%
    group_by(model) %>%
    summarise(
      intercept = coef(lm(y_pred ~ y_true))[1],
      slope = coef(lm(y_pred ~ y_true))[2],
      .groups = "drop"
    ) %>%
    mutate(
      label = sprintf("y = %.3fx %+ .3f", slope, intercept)
    )

  p_scatter <- ggplot(forecast_df, aes(x = y_true, y = y_pred)) +
    geom_abline(intercept = 0, slope = 1, colour = "#2A2A2A", linewidth = 0.55) +
    geom_point(
      colour = "#2F7FA6",
      size = 1.25,
      alpha = 0.78,
      stroke = 0
    ) +
    geom_smooth(
      method = "lm",
      formula = y ~ x,
      se = FALSE,
      colour = "#C23B22",
      linewidth = 0.75
    ) +
    geom_text(
      data = eqn_annot,
      aes(label = label),
      x = -Inf,
      y = Inf,
      inherit.aes = FALSE,
      hjust = -0.05,
      vjust = 1.25,
      size = 3.3,
      colour = "#C23B22",
      family = "Times New Roman"
    ) +
    facet_wrap(~model, ncol = 5, scales = "free") +
    scale_x_continuous(expand = expansion(mult = c(0.02, 0.04))) +
    scale_y_continuous(expand = expansion(mult = c(0.02, 0.04))) +
    labs(
      title = "图3-4 24小时预测时域预测值与真实值点云一致性对比",
      subtitle = "站点41010，同一测试窗口；黑线为理想一致性参考线，红线为线性拟合结果",
      x = "真实有效波高（米）",
      y = "预测有效波高（米）"
    ) +
    theme(
      legend.position = "none",
      strip.text = element_text(face = "plain", size = 10.5, colour = "#1D2D3A"),
      panel.spacing = unit(0.55, "lines"),
      panel.grid.major = element_line(colour = "#D9D9D9", linewidth = 0.28),
      panel.grid.minor = element_blank(),
      axis.text = element_text(size = 9),
      axis.title = element_text(size = 11, face = "plain")
    )

  save_dual(p_scatter, "fig4_prediction_scatter_24h.png", width = 13.2, height = 3.25)

  resid_df <- forecast_df %>%
    mutate(
      residual = y_pred - y_true,
      abs_residual = abs(residual)
    )

  p_resid <- ggplot(resid_df, aes(x = residual, fill = model, colour = model)) +
    geom_density(alpha = 0.18, linewidth = 0.8, adjust = 1.05) +
    geom_vline(xintercept = 0, colour = "#333333", linewidth = 0.7, linetype = "dashed") +
    facet_wrap(~model, ncol = 1, scales = "free_y") +
    scale_fill_manual(values = model_palette[names(model_palette) %in% unique(resid_df$model)]) +
    scale_colour_manual(values = model_palette[names(model_palette) %in% unique(resid_df$model)]) +
    labs(
      title = "A. 残差分布",
      subtitle = "残差定义为预测值减真实值；越集中于0附近，说明系统偏差越小",
      x = "残差（米）",
      y = "核密度"
    ) +
    theme(
      legend.position = "none",
      strip.text = element_text(face = "bold", size = 10.2),
      panel.spacing.y = unit(0.45, "lines")
    )

  p_calib <- ggplot(resid_df, aes(x = y_true, y = residual, colour = model)) +
    geom_hline(yintercept = 0, colour = "#333333", linewidth = 0.7, linetype = "dashed") +
    geom_point(alpha = 0.32, size = 1.0, stroke = 0) +
    geom_smooth(method = "loess", se = FALSE, linewidth = 0.8, span = 0.85) +
    facet_wrap(~model, ncol = 1) +
    scale_colour_manual(values = model_palette[names(model_palette) %in% unique(resid_df$model)]) +
    labs(
      title = "B. 残差校准关系",
      subtitle = "横轴为真实值，纵轴为残差；平滑曲线越贴近0线，说明校准越稳定",
      x = "真实有效波高（米）",
      y = "残差（米）"
    ) +
    theme(
      legend.position = "none",
      strip.text = element_text(face = "bold", size = 10.2),
      panel.spacing.y = unit(0.45, "lines")
    )

  p_diag <- plot_grid(
    p_resid,
    p_calib,
    ncol = 2,
    rel_widths = c(1, 1.08),
    labels = NULL,
    align = "h"
  )

  title_grob <- ggdraw() +
    draw_label(
      "图3-5 24小时预测时域残差分布与校准诊断图",
      x = 0.5, y = 0.68,
      fontfamily = "Microsoft YaHei",
      fontface = "bold",
      size = 15,
      colour = "#17324D"
    ) +
    draw_label(
      "同一测试窗口下比较五类模型误差分布与校准稳定性，作为时间曲线与点云图的补充证据",
      x = 0.5, y = 0.18,
      fontfamily = "Microsoft YaHei",
      size = 10.5,
      colour = "#516B84"
    )

  p_diag_full <- plot_grid(title_grob, p_diag, ncol = 1, rel_heights = c(0.11, 0.89))
  save_dual(p_diag_full, "fig5_residual_calibration_24h.png", width = 12.4, height = 9.2)
}

distill_path <- file.path(table_dir, "distill_all_horizons_metrics.csv")
if (file.exists(distill_path)) {
  distill_df <- read.csv(distill_path, check.names = FALSE)
  h_levels <- sort(unique(distill_df$horizon_hours))
  metric_levels <- c("MAE", "RMSE", "MASE", "SMAPE")
  metric_palette <- c(
    "MAE" = "#0E5A8A",
    "RMSE" = "#D97A2B",
    "MASE" = "#2E8B57",
    "SMAPE" = "#A54A7A"
  )

  baseline_df <- distill_df %>%
    filter(alpha == 1.0) %>%
    select(
      horizon_hours,
      MAE_base = MAE,
      RMSE_base = RMSE,
      MASE_base = MASE,
      SMAPE_base = SMAPE
    )

  joined <- distill_df %>%
    left_join(baseline_df, by = "horizon_hours")

  long_df <- bind_rows(
    data.frame(horizon_hours = joined$horizon_hours, alpha = joined$alpha, metric = "MAE", value = joined$MAE, baseline = joined$MAE_base),
    data.frame(horizon_hours = joined$horizon_hours, alpha = joined$alpha, metric = "RMSE", value = joined$RMSE, baseline = joined$RMSE_base),
    data.frame(horizon_hours = joined$horizon_hours, alpha = joined$alpha, metric = "MASE", value = joined$MASE, baseline = joined$MASE_base),
    data.frame(horizon_hours = joined$horizon_hours, alpha = joined$alpha, metric = "SMAPE", value = joined$SMAPE, baseline = joined$SMAPE_base)
  ) %>%
    mutate(
      metric = factor(metric, levels = metric_levels),
      horizon_label = factor(paste0(horizon_hours, "h"), levels = paste0(h_levels, "h")),
      improve_pct = pmax((baseline - value) / baseline * 100, 0),
      alpha_label = sprintf("α=%.1f", alpha)
    )

  best_long <- long_df %>%
    group_by(horizon_hours, metric) %>%
    arrange(value, alpha, .by_group = TRUE) %>%
    slice(1) %>%
    ungroup() %>%
    mutate(
      horizon_label = factor(paste0(horizon_hours, "h"), levels = paste0(h_levels, "h")),
      metric = factor(metric, levels = metric_levels)
    )

  p_gain <- ggplot(best_long, aes(x = horizon_label, y = improve_pct, fill = metric)) +
    geom_hline(yintercept = 0, colour = "#788A9A", linewidth = 0.35) +
    geom_col(width = 0.68, colour = "white", linewidth = 0.2, show.legend = FALSE) +
    geom_text(aes(label = alpha_label), vjust = -0.45, size = 3.15, colour = "#17324D") +
    facet_wrap(~metric, ncol = 2, scales = "free_y") +
    scale_fill_manual(values = metric_palette) +
    scale_y_continuous(expand = expansion(mult = c(0.04, 0.18)), labels = label_number(accuracy = 0.1, suffix = "%")) +
    labs(
      title = "A. 各预测时域最优蒸馏收益",
      subtitle = "以 alpha = 1.0 纯真实标签模型为对照组，柱高表示对应指标的最佳相对降幅",
      x = "预测时域",
      y = "最优相对降幅"
    ) +
    theme(
      legend.position = "none",
      strip.background = element_rect(fill = "#F4F8FB", colour = NA),
      panel.spacing = unit(0.9, "lines")
    )

  p_alpha <- ggplot(best_long, aes(x = horizon_label, y = metric, fill = alpha)) +
    geom_tile(colour = "white", linewidth = 0.7) +
    geom_text(aes(label = sprintf("%.1f", alpha)), size = 3.6, colour = "#17324D", fontface = "bold") +
    scale_fill_gradientn(
      colours = c("#0E5A8A", "#6AA8C8", "#EBD9A8", "#D97A2B", "#A53F2B"),
      limits = c(0, 1),
      breaks = c(0, 0.2, 0.4, 0.5, 1.0),
      labels = c("0.0", "0.2", "0.4", "0.5", "1.0")
    ) +
    labs(
      title = "B. 各指标最优 alpha 分布",
      subtitle = "颜色和数字共同表示不同预测时域下的最优标签混合权重",
      x = "预测时域",
      y = "评价指标",
      fill = "最优 α"
    ) +
    theme(
      legend.position = "top",
      panel.grid = element_blank()
    )

  save_dual(p_gain, "fig4_distill_vs_real_relative.png", width = 11.6, height = 7.8)

  absolute_df <- bind_rows(
    best_long %>%
      transmute(
        horizon_hours,
        horizon_label,
        metric,
        alpha,
        group = "baseline",
        metric_value = baseline
      ),
    best_long %>%
      transmute(
        horizon_hours,
        horizon_label,
        metric,
        alpha,
        group = "distilled",
        metric_value = value
      )
  ) %>%
    mutate(
      group = factor(
        group,
        levels = c("baseline", "distilled"),
        labels = c("真实标签对照组 α=1.0", "最佳蒸馏方案")
      )
    )

  annot_df <- best_long %>%
    transmute(
      horizon_label,
      metric,
      note = sprintf("最佳 α=%s", format(alpha, nsmall = 1))
    )

  group_palette <- c(
    "真实标签对照组 α=1.0" = "#C9D6E2",
    "最佳蒸馏方案" = "#0E5A8A"
  )

  p_abs <- ggplot(absolute_df, aes(x = horizon_label, y = metric_value, fill = group)) +
    geom_col(
      position = position_dodge(width = 0.76),
      width = 0.62,
      colour = "white",
      linewidth = 0.22
    ) +
    geom_text(
      data = annot_df,
      aes(x = horizon_label, y = Inf, label = note),
      inherit.aes = FALSE,
      vjust = 1.5,
      size = 3.0,
      colour = "#17324D"
    ) +
    facet_wrap(~metric, ncol = 2, scales = "free_y") +
    scale_fill_manual(values = group_palette) +
    scale_y_continuous(expand = expansion(mult = c(0.04, 0.18))) +
    labs(
      title = "图4-4 各预测时域最佳蒸馏方案与纯真实标签训练对照组的绝对指标比较",
      subtitle = "每个时域仅展示该指标下表现最优的蒸馏方案，与 alpha = 1.0 对照组成对比较",
      x = "预测时域",
      y = "指标值",
      fill = "方案"
    ) +
    theme(
      strip.background = element_rect(fill = "#F4F8FB", colour = NA),
      panel.spacing = unit(0.9, "lines")
    )

  save_dual(p_abs, "fig4_distill_vs_real_absolute.png", width = 11.8, height = 8.2)
}
