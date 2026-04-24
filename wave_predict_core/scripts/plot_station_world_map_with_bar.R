#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(ggplot2)
  library(dplyr)
  library(maps)
  library(cowplot)
  library(grid)
})

# Output path can be overridden from command line:
# Rscript scripts/plot_station_world_map_with_bar.R output.png
args <- commandArgs(trailingOnly = TRUE)
output_path <- if (length(args) >= 1) {
  args[1]
} else {
  file.path(
    "-2026-LaTeX--main",
    "figures",
    "from_111md",
    "fig_station_world_map_with_bar.png"
  )
}

stations <- tibble::tribble(
  ~station_id, ~region_name,    ~lon,      ~lat,    ~depth_m,
  "41010",     "亚热带大西洋",   -78.467, 28.878,   888.0,
  "41043",     "加勒比热带",     -64.864, 21.090,  5286.0,
  "42040",     "墨西哥湾",       -88.237, 29.207,   192.0,
  "44025",     "北大西洋陆架",   -73.175, 40.258,    40.2,
  "46026",     "加利福尼亚流",   -122.838, 37.750,    53.0,
  "46050",     "俄勒冈-华盛顿",  -124.535, 44.679,   149.0,
  "46061",     "阿拉斯加高纬",   -146.837, 60.230,   215.0,
  "51001",     "夏威夷开阔洋",   -162.030, 24.475, 4895.0
) %>%
  mutate(
    station_id = factor(station_id, levels = station_id),
    label_nudge_x = c(7, 7, -11, 7, -11, -11, -12, -13),
    label_nudge_y = c(4, -4, -4, 4, -4, 4, 4, -4)
  )

world_map <- map_data("world")

base_family <- "sans"
map_fill <- "#f4efe6"
land_border <- "#b7aa92"
point_fill <- "#cc6b3d"
point_stroke <- "#4f2d1f"
bar_fill <- "#2f6c8f"
grid_color <- "#d8d8d8"

map_plot <- ggplot() +
  geom_polygon(
    data = world_map,
    aes(x = long, y = lat, group = group),
    fill = map_fill,
    color = land_border,
    linewidth = 0.22
  ) +
  geom_hline(
    yintercept = seq(-60, 60, by = 30),
    color = grid_color,
    linewidth = 0.2,
    linetype = "dashed"
  ) +
  geom_vline(
    xintercept = seq(-150, 150, by = 60),
    color = grid_color,
    linewidth = 0.2,
    linetype = "dashed"
  ) +
  geom_point(
    data = stations,
    aes(x = lon, y = lat),
    shape = 21,
    size = 3.6,
    stroke = 0.8,
    fill = point_fill,
    color = point_stroke
  ) +
  geom_curve(
    data = stations,
    aes(
      x = lon,
      y = lat,
      xend = lon + label_nudge_x,
      yend = lat + label_nudge_y
    ),
    curvature = 0.12,
    linewidth = 0.25,
    color = point_stroke,
    arrow = arrow(length = unit(0.08, "inches"), type = "closed")
  ) +
  geom_label(
    data = stations,
    aes(
      x = lon + label_nudge_x,
      y = lat + label_nudge_y,
      label = station_id
    ),
    family = base_family,
    size = 3.2,
    linewidth = 0.15,
    label.padding = unit(0.12, "lines"),
    fill = "white",
    color = "#202020"
  ) +
  coord_quickmap(xlim = c(-180, 20), ylim = c(-5, 75), expand = FALSE) +
  labs(
    title = "8个NDBC浮标站点全球位置分布",
    subtitle = "覆盖大西洋、墨西哥湾、美国西海岸、阿拉斯加及夏威夷海域",
    x = NULL,
    y = NULL
  ) +
  theme_minimal(base_family = base_family) +
  theme(
    plot.background = element_rect(fill = "white", color = NA),
    panel.background = element_rect(fill = "#fbfaf7", color = NA),
    panel.grid = element_blank(),
    axis.text = element_blank(),
    title = element_text(color = "#1f1f1f"),
    plot.title = element_text(size = 15, face = "bold"),
    plot.subtitle = element_text(size = 9.8, color = "#555555"),
    plot.margin = margin(10, 8, 8, 8)
  )

bar_plot <- stations %>%
  mutate(station_id = reorder(as.character(station_id), depth_m)) %>%
  ggplot(aes(x = station_id, y = depth_m)) +
  geom_col(width = 0.72, fill = bar_fill) +
  geom_text(
    aes(label = ifelse(depth_m >= 1000, sprintf("%.0f", depth_m), sprintf("%.1f", depth_m))),
    hjust = -0.08,
    size = 3.0,
    family = base_family,
    color = "#1f1f1f"
  ) +
  scale_y_continuous(expand = expansion(mult = c(0, 0.12))) +
  coord_flip() +
  labs(
    title = "站点水深",
    subtitle = "单位：m",
    x = NULL,
    y = NULL
  ) +
  theme_minimal(base_family = base_family) +
  theme(
    plot.background = element_rect(fill = "white", color = NA),
    panel.background = element_rect(fill = "white", color = NA),
    panel.grid.major.y = element_blank(),
    panel.grid.minor = element_blank(),
    panel.grid.major.x = element_line(color = "#e9e9e9", linewidth = 0.25),
    axis.text.y = element_text(size = 9, color = "#202020"),
    axis.text.x = element_blank(),
    axis.ticks = element_blank(),
    title = element_text(color = "#1f1f1f"),
    plot.title = element_text(size = 12, face = "bold"),
    plot.subtitle = element_text(size = 9, color = "#555555"),
    plot.margin = margin(16, 10, 12, 0)
  )

combined_plot <- plot_grid(
  map_plot,
  bar_plot,
  nrow = 1,
  rel_widths = c(2.9, 1.2),
  align = "h"
)

dir.create(dirname(output_path), recursive = TRUE, showWarnings = FALSE)
ggsave(
  filename = output_path,
  plot = combined_plot,
  width = 13.5,
  height = 6.8,
  dpi = 320,
  bg = "white"
)

message("Saved plot to: ", normalizePath(output_path, winslash = "/", mustWork = FALSE))
