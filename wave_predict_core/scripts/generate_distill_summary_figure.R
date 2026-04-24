#!/usr/bin/env Rscript

options(stringsAsFactors = FALSE)

local_lib <- normalizePath(file.path(getwd(), ".r_libs"), winslash = "/", mustWork = FALSE)
if (dir.exists(local_lib)) {
  .libPaths(c(local_lib, .libPaths()))
}

suppressPackageStartupMessages({
  library(ggplot2)
  library(dplyr)
  library(scales)
  library(grid)
})

if (.Platform$OS.type == "windows" && file.exists("C:/Windows/Fonts/simhei.ttf")) {
  grDevices::windowsFonts(SimHei = grDevices::windowsFont("SimHei"))
}
font_family <- if (.Platform$OS.type == "windows" && "SimHei" %in% names(grDevices::windowsFonts())) {
  "SimHei"
} else {
  "sans"
}

root_dir <- normalizePath(getwd(), winslash = "/", mustWork = TRUE)
table_dir <- file.path(root_dir, "thesis_assets", "tables")
fig_dir <- file.path(root_dir, "thesis_assets", "figures")
latex_fig_dir <- file.path(root_dir, "-2026-LaTeX--main", "figures", "from_111md")

dir.create(fig_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(latex_fig_dir, recursive = TRUE, showWarnings = FALSE)

theme_set(
  theme_minimal(base_family = font_family, base_size = 12) +
    theme(
      plot.title = element_text(face = "bold", size = 14, colour = "#17324D"),
      plot.subtitle = element_text(size = 10.5, colour = "#516B84"),
      axis.title = element_text(face = "bold", colour = "#25435E"),
      axis.text = element_text(colour = "#35556F"),
      legend.title = element_text(face = "bold"),
      legend.position = "top",
      panel.grid.minor = element_blank(),
      panel.grid.major = element_line(colour = "#DCE6EE", linewidth = 0.35),
      strip.text = element_text(face = "bold", colour = "#17324D"),
      plot.background = element_rect(fill = "white", colour = NA),
      panel.background = element_rect(fill = "white", colour = NA)
    )
)

save_dual <- function(plot_obj, filename, width, height, dpi = 320, bg = "white") {
  out1 <- file.path(fig_dir, filename)
  out2 <- file.path(latex_fig_dir, filename)
  ggsave(out1, plot_obj, width = width, height = height, dpi = dpi, bg = bg)
  file.copy(out1, out2, overwrite = TRUE)
}

distill_path <- file.path(table_dir, "distill_all_horizons_metrics.csv")
if (!file.exists(distill_path)) {
  stop("missing file: ", distill_path)
}

distill_df <- read.csv(distill_path, check.names = FALSE)
metric_levels <- c("MAE", "RMSE", "MASE", "SMAPE")
h_levels <- sort(unique(distill_df$horizon_hours))
h_labels <- paste0(h_levels, "h")
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

candidate_df <- distill_df %>%
  filter(alpha < 1.0) %>%
  left_join(baseline_df, by = "horizon_hours")

long_df <- bind_rows(
  data.frame(horizon_hours = candidate_df$horizon_hours, alpha = candidate_df$alpha, metric = "MAE", value = candidate_df$MAE, baseline = candidate_df$MAE_base),
  data.frame(horizon_hours = candidate_df$horizon_hours, alpha = candidate_df$alpha, metric = "RMSE", value = candidate_df$RMSE, baseline = candidate_df$RMSE_base),
  data.frame(horizon_hours = candidate_df$horizon_hours, alpha = candidate_df$alpha, metric = "MASE", value = candidate_df$MASE, baseline = candidate_df$MASE_base),
  data.frame(horizon_hours = candidate_df$horizon_hours, alpha = candidate_df$alpha, metric = "SMAPE", value = candidate_df$SMAPE, baseline = candidate_df$SMAPE_base)
) %>%
  mutate(
    metric = factor(metric, levels = metric_levels),
    horizon_label = factor(paste0(horizon_hours, "h"), levels = h_labels),
    relative_change_pct = (baseline - value) / baseline * 100,
    alpha_label = sprintf("α=%.1f", alpha)
  )

best_by_metric <- long_df %>%
  group_by(horizon_hours, metric) %>%
  arrange(value, alpha, .by_group = TRUE) %>%
  slice(1) %>%
  ungroup() %>%
  mutate(
    horizon_label = factor(paste0(horizon_hours, "h"), levels = h_labels),
    label = sprintf("%.2f%%", relative_change_pct)
  )

p_rel <- ggplot(best_by_metric, aes(x = horizon_label, y = relative_change_pct, fill = metric)) +
  geom_hline(yintercept = 0, colour = "#6D7F8F", linewidth = 0.4) +
  geom_col(width = 0.68, colour = "white", linewidth = 0.22, show.legend = FALSE) +
  geom_text(
    aes(label = alpha_label),
    vjust = ifelse(best_by_metric$relative_change_pct >= 0, -0.55, 1.25),
    size = 3.0,
    colour = "#17324D"
  ) +
  facet_wrap(~metric, ncol = 2, scales = "free_y") +
  scale_fill_manual(values = metric_palette) +
  scale_y_continuous(
    labels = label_number(accuracy = 0.1, suffix = "%"),
    expand = expansion(mult = c(0.12, 0.18))
  ) +
  labs(
    title = "图4-3 各预测时域最佳蒸馏方案相对真实标签对照组的变化幅度",
    subtitle = "对照组固定为 alpha = 1.0；正值表示蒸馏后误差下降，负值表示相对纯真实标签训练退化",
    x = "预测时域",
    y = "相对变化幅度"
  ) +
  theme(
    strip.background = element_rect(fill = "#F4F8FB", colour = NA),
    panel.spacing = unit(0.9, "lines")
  )

save_dual(p_rel, "fig4_distill_vs_real_relative.png", width = 11.6, height = 7.8)

absolute_df <- best_by_metric %>%
  transmute(
    horizon_hours,
    horizon_label,
    metric,
    alpha,
    group = "distilled",
    metric_value = value
  ) %>%
  bind_rows(
    best_by_metric %>%
      transmute(
        horizon_hours,
        horizon_label,
        metric,
        alpha,
        group = "baseline",
        metric_value = baseline
      )
  ) %>%
  mutate(
    group = factor(
      group,
      levels = c("baseline", "distilled"),
      labels = c("真实标签对照组 α=1.0", "最佳蒸馏方案")
    ),
    metric = factor(metric, levels = metric_levels)
  )

annot_df <- best_by_metric %>%
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

line_palette <- c(
  "3h" = "#12355B",
  "6h" = "#1D6F8F",
  "12h" = "#2A9D8F",
  "24h" = "#76B041",
  "48h" = "#F2A541",
  "72h" = "#D95D39",
  "120h" = "#7A3E65"
)

line_base_df <- distill_df %>%
  left_join(baseline_df, by = "horizon_hours")

line_df <- bind_rows(
  data.frame(horizon_hours = line_base_df$horizon_hours, alpha = line_base_df$alpha, metric = "MAE", value = line_base_df$MAE, baseline = line_base_df$MAE_base),
  data.frame(horizon_hours = line_base_df$horizon_hours, alpha = line_base_df$alpha, metric = "RMSE", value = line_base_df$RMSE, baseline = line_base_df$RMSE_base),
  data.frame(horizon_hours = line_base_df$horizon_hours, alpha = line_base_df$alpha, metric = "MASE", value = line_base_df$MASE, baseline = line_base_df$MASE_base),
  data.frame(horizon_hours = line_base_df$horizon_hours, alpha = line_base_df$alpha, metric = "SMAPE", value = line_base_df$SMAPE, baseline = line_base_df$SMAPE_base)
) %>%
  mutate(
    metric = factor(metric, levels = metric_levels),
    horizon_label = factor(paste0(horizon_hours, "h"), levels = h_labels),
    relative_change_pct = (baseline - value) / baseline * 100
  )

best_points <- line_df %>%
  group_by(horizon_hours, metric) %>%
  arrange(value, alpha, .by_group = TRUE) %>%
  slice(1) %>%
  ungroup() %>%
  mutate(
    horizon_label = factor(paste0(horizon_hours, "h"), levels = h_labels),
    point_label = sprintf("α=%.1f", alpha)
  )

p_alpha_curve <- ggplot(
  line_df,
  aes(x = alpha, y = relative_change_pct, colour = horizon_label, group = horizon_label)
) +
  geom_hline(yintercept = 0, colour = "#4F6172", linewidth = 0.45, linetype = "dashed") +
  geom_line(linewidth = 0.85, alpha = 0.94) +
  geom_point(size = 2.05, alpha = 0.96) +
  geom_point(
    data = best_points,
    aes(x = alpha, y = relative_change_pct),
    inherit.aes = FALSE,
    shape = 21,
    size = 3.1,
    stroke = 0.65,
    fill = "white",
    colour = "#151F2B"
  ) +
  facet_wrap(~metric, ncol = 2, scales = "free_y") +
  scale_colour_manual(values = line_palette, name = "预测时域") +
  scale_x_continuous(
    breaks = c(0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0),
    labels = c("0", "0.1", "0.2", "0.3", "0.4", "0.5", "0.7", "1.0")
  ) +
  scale_y_continuous(labels = label_number(accuracy = 0.1, suffix = "%")) +
  labs(
    title = "图4-5 不同预测时域下蒸馏权重 α 的误差响应曲线",
    subtitle = "纵轴为相对 alpha = 1.0 纯真实标签对照组的误差变化；正值表示误差下降，空心点标出各曲线最优权重",
    x = "真实标签混合权重 α",
    y = "相对误差变化"
  ) +
  theme(
    legend.position = "top",
    legend.key.width = unit(1.25, "lines"),
    strip.background = element_rect(fill = "#F4F8FB", colour = NA),
    panel.spacing = unit(0.95, "lines"),
    panel.grid.major.x = element_line(colour = "#E5EDF4", linewidth = 0.28)
  )

save_dual(p_alpha_curve, "fig4_distill_alpha_response_lines.png", width = 12.2, height = 8.3)
