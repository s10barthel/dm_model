# ============================================================
# Pass-score validity analysis against coach ratings
# ============================================================
# This script implements the revised data analysis plan:
# 1. ICC(2,6) for reliability of the aggregated coach reference
# 2. Linear mixed models for score-level validity
# 3. Calibration and residual plots
# 4. Within/between-scene decomposition as a supplementary analysis
# 5. Ranking validity in the full dataset only
# 6. Top-selection validity in the full dataset only
#
# Expected setup:
# - coach_ratings.csv is stored in the same folder as this script
# - results are written to a subfolder named "results"
# ============================================================

# ----------------------------
# 0) User settings
# ----------------------------
MODEL_SCORE_VAR <- "pass_score_max"   # Change here if needed:
                                      # "pass_score_max", "pass_score_avg",
                                      # "pass_score_med", "pass_score_br"

HIGH_INTENT_THRESHOLD <- 0.05
RBO_P <- 0.5
N_BOOT <- 2000
SEED <- 12345
INSTALL_MISSING_PACKAGES <- FALSE

# Optional explicit Windows base directory from the project note.
# The script uses the directory of the script file by default.
WINDOWS_BASE_DIR <- "C:/Users/steffen.barthel/OneDrive - TSG 1899 Hoffenheim Fußball-Spielbetriebs GmbH/Dokumente/VS Code/.vscode/dm_model/validation/coach_ratings"

# ----------------------------
# 1) Package setup
# ----------------------------
required_packages <- c(
  "readr",
  "dplyr",
  "tidyr",
  "purrr",
  "tibble",
  "stringr",
  "ggplot2",
  "lmerTest",
  "performance",
  "irr"
)

install_and_load <- function(pkgs, install_missing = FALSE) {
  missing_pkgs <- pkgs[!vapply(pkgs, requireNamespace, quietly = TRUE, FUN.VALUE = logical(1))]

  if (length(missing_pkgs) > 0) {
    msg <- paste0(
      "Missing packages: ", paste(missing_pkgs, collapse = ", "), "."
    )
    if (!install_missing) {
      stop(
        msg,
        "\nSet INSTALL_MISSING_PACKAGES <- TRUE at the top of the script,",
        "\nor install them manually before running the script."
      )
    }
    install.packages(missing_pkgs)
  }

  invisible(lapply(pkgs, library, character.only = TRUE))
}

install_and_load(required_packages, install_missing = INSTALL_MISSING_PACKAGES)

# ----------------------------
# 2) Helper functions
# ----------------------------
get_script_dir <- function() {
  cmd_args <- commandArgs(trailingOnly = FALSE)
  file_arg <- "--file="
  path_from_args <- sub(file_arg, "", cmd_args[grep(file_arg, cmd_args)])

  if (length(path_from_args) > 0) {
    return(dirname(normalizePath(path_from_args[1], winslash = "/", mustWork = FALSE)))
  }

  if (!is.null(sys.frames()[[1]]$ofile)) {
    return(dirname(normalizePath(sys.frames()[[1]]$ofile, winslash = "/", mustWork = FALSE)))
  }

  # Fallback for interactive use
  getwd()
}

safe_dir_create <- function(path) {
  if (!dir.exists(path)) {
    dir.create(path, recursive = TRUE, showWarnings = FALSE)
  }
}

rmse_vec <- function(obs, pred) {
  sqrt(mean((obs - pred)^2, na.rm = TRUE))
}

mae_vec <- function(obs, pred) {
  mean(abs(obs - pred), na.rm = TRUE)
}

bootstrap_metric_summary <- function(x, metric_name, n_boot = 2000, seed = 12345) {
  x <- x[is.finite(x)]
  n <- length(x)

  if (n == 0) {
    return(tibble::tibble(
      metric = metric_name,
      statistic = c("mean", "median"),
      n_scenes = 0,
      estimate = NA_real_,
      ci_low = NA_real_,
      ci_high = NA_real_
    ))
  }

  set.seed(seed)
  boot_means <- replicate(n_boot, mean(sample(x, size = n, replace = TRUE), na.rm = TRUE))
  boot_medians <- replicate(n_boot, median(sample(x, size = n, replace = TRUE), na.rm = TRUE))

  tibble::tibble(
    metric = metric_name,
    statistic = c("mean", "median"),
    n_scenes = n,
    estimate = c(mean(x, na.rm = TRUE), median(x, na.rm = TRUE)),
    ci_low = c(
      unname(stats::quantile(boot_means, probs = 0.025, na.rm = TRUE)),
      unname(stats::quantile(boot_medians, probs = 0.025, na.rm = TRUE))
    ),
    ci_high = c(
      unname(stats::quantile(boot_means, probs = 0.975, na.rm = TRUE)),
      unname(stats::quantile(boot_medians, probs = 0.975, na.rm = TRUE))
    )
  )
}

kendall_tau_b <- function(x, y) {
  ok <- stats::complete.cases(x, y)
  x <- x[ok]
  y <- y[ok]

  n <- length(x)
  if (n < 2) {
    return(NA_real_)
  }

  cmb <- utils::combn(n, 2)
  dx <- x[cmb[1, ]] - x[cmb[2, ]]
  dy <- y[cmb[1, ]] - y[cmb[2, ]]

  concordant <- sum(dx * dy > 0)
  discordant <- sum(dx * dy < 0)
  ties_x <- sum(dx == 0 & dy != 0)
  ties_y <- sum(dy == 0 & dx != 0)

  denom <- sqrt((concordant + discordant + ties_x) * (concordant + discordant + ties_y))
  if (denom == 0) {
    return(NA_real_)
  }

  (concordant - discordant) / denom
}

make_rank_min_desc <- function(x) {
  dplyr::min_rank(dplyr::desc(x))
}

top_k_with_ties <- function(x, k = 3) {
  ranks <- make_rank_min_desc(x)
  which(ranks <= k)
}

jaccard_similarity <- function(a, b) {
  if (length(a) == 0 && length(b) == 0) return(NA_real_)
  union_size <- length(union(a, b))
  if (union_size == 0) return(NA_real_)
  length(intersect(a, b)) / union_size
}

overlap_at_k <- function(a, b) {
  denom <- min(length(a), length(b))
  if (denom == 0) return(NA_real_)
  length(intersect(a, b)) / denom
}

# Tie-aware RBO implementation:
# For a tie group that spans multiple positions, the positional influence
# is split equally across the tied items. At each depth d, if a tie group
# is only partially included, each item in that tie group receives a
# fractional inclusion weight.
build_tie_groups <- function(scores, item_ids) {
  ord <- order(scores, decreasing = TRUE, na.last = NA)
  scores_sorted <- scores[ord]
  items_sorted <- item_ids[ord]

  split_items <- split(items_sorted, scores_sorted)
  unique_scores_desc <- sort(unique(scores_sorted), decreasing = TRUE)
  split_items <- split_items[as.character(unique_scores_desc)]

  start_pos <- integer(length(split_items))
  end_pos <- integer(length(split_items))
  sizes <- integer(length(split_items))
  cursor <- 1L

  for (i in seq_along(split_items)) {
    g <- length(split_items[[i]])
    start_pos[i] <- cursor
    end_pos[i] <- cursor + g - 1L
    sizes[i] <- g
    cursor <- cursor + g
  }

  list(
    groups = split_items,
    start_pos = start_pos,
    end_pos = end_pos,
    sizes = sizes,
    n_items = length(items_sorted)
  )
}

prefix_membership <- function(tie_info, depth) {
  ids <- unlist(tie_info$groups, use.names = FALSE)
  memberships <- stats::setNames(rep(0, length(ids)), ids)

  for (i in seq_along(tie_info$groups)) {
    grp_ids <- tie_info$groups[[i]]
    s <- tie_info$start_pos[i]
    e <- tie_info$end_pos[i]
    g <- tie_info$sizes[i]

    if (depth < s) {
      frac <- 0
    } else if (depth >= e) {
      frac <- 1
    } else {
      frac <- (depth - s + 1) / g
    }
    memberships[grp_ids] <- frac
  }

  memberships
}

rbo_tie_adjusted <- function(scores_a, scores_b, item_ids, p = 0.5) {
  ok <- stats::complete.cases(scores_a, scores_b, item_ids)
  scores_a <- scores_a[ok]
  scores_b <- scores_b[ok]
  item_ids <- item_ids[ok]

  n <- length(item_ids)
  if (n < 1) {
    return(NA_real_)
  }

  tie_a <- build_tie_groups(scores_a, item_ids)
  tie_b <- build_tie_groups(scores_b, item_ids)

  overlaps_at_depth <- numeric(n)

  for (d in seq_len(n)) {
    mem_a <- prefix_membership(tie_a, d)
    mem_b <- prefix_membership(tie_b, d)

    all_ids <- union(names(mem_a), names(mem_b))
    vec_a <- mem_a[all_ids]
    vec_b <- mem_b[all_ids]
    vec_a[is.na(vec_a)] <- 0
    vec_b[is.na(vec_b)] <- 0

    overlap_size <- sum(pmin(vec_a, vec_b))
    overlaps_at_depth[d] <- overlap_size / d
  }

  # Extrapolated finite-list RBO
  weighted_prefix <- (1 - p) * sum((p ^ (seq_len(n) - 1)) * overlaps_at_depth)
  tail_term <- (p ^ n) * overlaps_at_depth[n]
  weighted_prefix + tail_term
}

summarise_scene_sizes <- function(dat, dataset_name) {
  if (nrow(dat) == 0) {
    return(tibble::tibble(
      dataset = dataset_name,
      n_rows = 0,
      n_scenes = 0,
      min_scene_size = NA_real_,
      q1_scene_size = NA_real_,
      median_scene_size = NA_real_,
      mean_scene_size = NA_real_,
      q3_scene_size = NA_real_,
      max_scene_size = NA_real_
    ))
  }

  scene_sizes <- dat %>%
    dplyr::count(SceneNr_original, name = "n_options")

  tibble::tibble(
    dataset = dataset_name,
    n_rows = nrow(dat),
    n_scenes = dplyr::n_distinct(dat$SceneNr_original),
    min_scene_size = min(scene_sizes$n_options),
    q1_scene_size = unname(stats::quantile(scene_sizes$n_options, 0.25)),
    median_scene_size = stats::median(scene_sizes$n_options),
    mean_scene_size = mean(scene_sizes$n_options),
    q3_scene_size = unname(stats::quantile(scene_sizes$n_options, 0.75)),
    max_scene_size = max(scene_sizes$n_options)
  )
}

fit_main_lmm <- function(dat, dataset_label, results_dir) {
  model <- lmerTest::lmer(
    Scores ~ model_score_z + (1 | SceneNr_original),
    data = dat,
    REML = TRUE
  )

  coef_table <- as.data.frame(summary(model)$coefficients)
  coef_table$term <- rownames(coef_table)
  rownames(coef_table) <- NULL

  ci <- as.data.frame(stats::confint(model, method = "Wald"))
  ci$term <- rownames(ci)
  rownames(ci) <- NULL
  names(ci)[1:2] <- c("conf_low", "conf_high")

  fixed_effects <- coef_table %>%
    dplyr::left_join(ci, by = "term") %>%
    dplyr::select(term, Estimate, `Std. Error`, df, `t value`, `Pr(>|t|)`, conf_low, conf_high)

  predictions <- dat %>%
    dplyr::mutate(
      fitted = as.numeric(stats::fitted(model)),
      residual = as.numeric(stats::residuals(model))
    )

  r2_vals <- performance::r2_nakagawa(model)
  model_fit <- tibble::tibble(
    dataset = dataset_label,
    n_rows = nrow(dat),
    n_scenes = dplyr::n_distinct(dat$SceneNr_original),
    rmse = rmse_vec(predictions$Scores, predictions$fitted),
    mae = mae_vec(predictions$Scores, predictions$fitted),
    marginal_r2 = unname(r2_vals$R2_marginal),
    conditional_r2 = unname(r2_vals$R2_conditional)
  )

  cal_plot <- ggplot2::ggplot(predictions, ggplot2::aes(x = fitted, y = Scores)) +
    ggplot2::geom_point(alpha = 0.5) +
    ggplot2::geom_abline(slope = 1, intercept = 0, linetype = "dashed") +
    ggplot2::geom_smooth(method = "loess", se = TRUE) +
    ggplot2::labs(
      title = paste0("Calibration plot: ", dataset_label),
      x = "Predicted coach Score",
      y = "Observed coach Score"
    ) +
    ggplot2::theme_minimal(base_size = 12)

  resid_plot <- ggplot2::ggplot(predictions, ggplot2::aes(x = fitted, y = residual)) +
    ggplot2::geom_point(alpha = 0.5) +
    ggplot2::geom_hline(yintercept = 0, linetype = "dashed") +
    ggplot2::geom_smooth(method = "loess", se = TRUE) +
    ggplot2::labs(
      title = paste0("Residual plot: ", dataset_label),
      x = "Fitted value",
      y = "Residual"
    ) +
    ggplot2::theme_minimal(base_size = 12)

  readr::write_csv(fixed_effects, file.path(results_dir, paste0("lmm_", dataset_label, "_fixed_effects.csv")))
  readr::write_csv(model_fit, file.path(results_dir, paste0("lmm_", dataset_label, "_model_fit.csv")))
  readr::write_csv(predictions, file.path(results_dir, paste0("lmm_", dataset_label, "_predictions.csv")))
  ggplot2::ggsave(
    filename = file.path(results_dir, paste0("lmm_", dataset_label, "_calibration_plot.png")),
    plot = cal_plot, width = 8, height = 6, dpi = 300
  )
  ggplot2::ggsave(
    filename = file.path(results_dir, paste0("lmm_", dataset_label, "_residual_plot.png")),
    plot = resid_plot, width = 8, height = 6, dpi = 300
  )

  list(model = model, fixed_effects = fixed_effects, model_fit = model_fit, predictions = predictions)
}

fit_within_between_lmm <- function(dat, dataset_label, results_dir) {
  dat_wb <- dat %>%
    dplyr::group_by(SceneNr_original) %>%
    dplyr::mutate(
      model_score_z_between = mean(model_score_z, na.rm = TRUE),
      model_score_z_within = model_score_z - model_score_z_between
    ) %>%
    dplyr::ungroup()

  model <- lmerTest::lmer(
    Scores ~ model_score_z_within + model_score_z_between + (1 | SceneNr_original),
    data = dat_wb,
    REML = TRUE
  )

  coef_table <- as.data.frame(summary(model)$coefficients)
  coef_table$term <- rownames(coef_table)
  rownames(coef_table) <- NULL

  ci <- as.data.frame(stats::confint(model, method = "Wald"))
  ci$term <- rownames(ci)
  rownames(ci) <- NULL
  names(ci)[1:2] <- c("conf_low", "conf_high")

  fixed_effects <- coef_table %>%
    dplyr::left_join(ci, by = "term") %>%
    dplyr::select(term, Estimate, `Std. Error`, df, `t value`, `Pr(>|t|)`, conf_low, conf_high)

  predictions <- dat_wb %>%
    dplyr::mutate(
      fitted = as.numeric(stats::fitted(model)),
      residual = as.numeric(stats::residuals(model))
    )

  r2_vals <- performance::r2_nakagawa(model)
  model_fit <- tibble::tibble(
    dataset = dataset_label,
    n_rows = nrow(dat_wb),
    n_scenes = dplyr::n_distinct(dat_wb$SceneNr_original),
    rmse = rmse_vec(predictions$Scores, predictions$fitted),
    mae = mae_vec(predictions$Scores, predictions$fitted),
    marginal_r2 = unname(r2_vals$R2_marginal),
    conditional_r2 = unname(r2_vals$R2_conditional)
  )

  readr::write_csv(
    fixed_effects,
    file.path(results_dir, paste0("supplementary_lmm_within_between_", dataset_label, "_fixed_effects.csv"))
  )
  readr::write_csv(
    model_fit,
    file.path(results_dir, paste0("supplementary_lmm_within_between_", dataset_label, "_model_fit.csv"))
  )
  readr::write_csv(
    predictions,
    file.path(results_dir, paste0("supplementary_lmm_within_between_", dataset_label, "_predictions.csv"))
  )

  list(model = model, fixed_effects = fixed_effects, model_fit = model_fit, predictions = predictions)
}

ranking_metrics_by_scene <- function(dat, model_score_var, rbo_p = 0.5) {
  dat %>%
    dplyr::group_by(SceneNr_original) %>%
    dplyr::group_modify(~{
      scene_df <- .x %>%
        dplyr::mutate(
          scene_item_id = paste0(.y$SceneNr_original[[1]], "__", dplyr::row_number()),
          coach_rank_recomputed = make_rank_min_desc(Scores),
          model_rank_recomputed = make_rank_min_desc(.data[[model_score_var]])
        )

      coach_top3_ids <- scene_df$scene_item_id[top_k_with_ties(scene_df$Scores, k = 3)]
      model_top3_ids <- scene_df$scene_item_id[top_k_with_ties(scene_df[[model_score_var]], k = 3)]

      tibble::tibble(
        n_options = nrow(scene_df),
        tau_b = kendall_tau_b(scene_df$Scores, scene_df[[model_score_var]]),
        rbo_p_0_5 = rbo_tie_adjusted(
          scores_a = scene_df$Scores,
          scores_b = scene_df[[model_score_var]],
          item_ids = scene_df$scene_item_id,
          p = rbo_p
        ),
        coach_top3_size = length(coach_top3_ids),
        model_top3_size = length(model_top3_ids),
        jaccard_top3 = jaccard_similarity(coach_top3_ids, model_top3_ids),
        overlap_at3 = overlap_at_k(coach_top3_ids, model_top3_ids)
      )
    }) %>%
    dplyr::ungroup()
}

make_ranking_output_rows <- function(dat, model_score_var) {
  dat %>%
    dplyr::group_by(SceneNr_original) %>%
    dplyr::mutate(
      scene_row_id = dplyr::row_number(),
      coach_rank_recomputed = make_rank_min_desc(Scores),
      model_rank_recomputed = make_rank_min_desc(.data[[model_score_var]]),
      coach_top3_flag = coach_rank_recomputed <= 3,
      model_top3_flag = model_rank_recomputed <= 3
    ) %>%
    dplyr::ungroup()
}

write_readme <- function(results_dir, model_score_var, high_intent_threshold, rbo_p, n_boot) {
  readme_lines <- c(
    "# Pass-score validity analysis: result guide",
    "",
    "This folder contains all outputs generated by the R script for evaluating the criterion validity of the selected model score against coach `Scores`.",
    "",
    paste0("- Selected model score variable: `", model_score_var, "`"),
    paste0("- High-intent threshold: `action_intent >= ", high_intent_threshold, "`"),
    paste0("- RBO parameter: `p = ", rbo_p, "`"),
    paste0("- Bootstrap iterations for optional confidence intervals: `", n_boot, "`"),
    "",
    "## Core interpretation logic",
    "",
    "The analysis has four main goals:",
    "1. Determine whether the aggregated coach rating is a reliable expert reference standard.",
    "2. Test whether the selected model score predicts coach Scores on the original 1 to 7 scale.",
    "3. Test whether the model reproduces the coaches' within-scene ordering of passing options.",
    "4. Test whether the model identifies the same top passing options as the coaches.",
    "",
    "## Most important files for the main study aim",
    "",
    "### 1) `icc_2_6_results.csv`",
    "This file contains the reliability analysis of the six coach ratings.",
    "",
    "Vital information:",
    "- `icc_value`: reliability of the average of the six coaches",
    "- `ci_low_95`, `ci_high_95`: 95% confidence interval",
    "",
    "Interpretation:",
    "- Higher ICC means the aggregated coach score is more stable.",
    "- Values around 0.75 or higher are usually interpreted as good to excellent reliability.",
    "- If ICC is low, the coach reference itself is unstable, which weakens the validity argument for the model.",
    "",
    "### 2) `lmm_full_fixed_effects.csv`",
    "This file contains the main mixed-model fixed effects for the full dataset.",
    "",
    "Vital information:",
    "- `term == model_score_z`",
    "- `Estimate`: expected change in coach Score for a one SD increase in the selected model score",
    "- `conf_low`, `conf_high`: 95% Wald confidence interval",
    "- `Pr(>|t|)`: statistical evidence for a non-zero slope",
    "",
    "Interpretation:",
    "- A positive slope means higher model scores are associated with higher coach Scores.",
    "- A larger positive slope indicates stronger score-level criterion validity.",
    "- If the confidence interval includes 0, evidence for a score-level relationship is weak.",
    "",
    "### 3) `lmm_full_model_fit.csv`",
    "This file contains the model fit statistics for the full dataset.",
    "",
    "Vital information:",
    "- `marginal_r2`: variance explained by the model score alone",
    "- `conditional_r2`: variance explained by the model score plus scene clustering",
    "- `rmse`, `mae`: average prediction error on the coach 1 to 7 scale",
    "",
    "Interpretation:",
    "- Higher marginal R2 is better because it means the selected model score explains more variance in coach Scores.",
    "- Lower RMSE and MAE are better because they indicate smaller average prediction errors.",
    "- If conditional R2 is much larger than marginal R2, scene context explains substantial additional variance.",
    "",
    "### 4) `lmm_high_intent_fixed_effects.csv` and `lmm_high_intent_model_fit.csv`",
    "These files repeat the score-level mixed model after filtering to higher-intent options.",
    "",
    "Interpretation:",
    "- Compare these outputs with the full dataset to see whether the model performs differently when very low-intent options are removed.",
    "- This is a secondary sensitivity analysis, not a replacement for the full-dataset analysis.",
    "",
    "### 5) `full_dataset_ranking_metric_summary.csv`",
    "This file contains the ranking-validity summaries in the full dataset only.",
    "",
    "Vital information:",
    "- `tau_b`: Kendall's tau-b across scenes",
    "- `rbo_p_0_5`: tie-adjusted Rank-Biased Overlap across scenes",
    "- `jaccard_top3`: Top-3 set similarity",
    "- `overlap_at3`: proportion of shared Top-3 players relative to the smaller Top-3 set",
    "- For each metric, both the mean and median are reported, together with bootstrap 95% confidence intervals.",
    "",
    "Interpretation:",
    "- Higher values are better for all four metrics.",
    "- `tau_b` close to 1 means the model reproduces the coaches' relative ordering within scenes very well.",
    "- `rbo_p_0_5` gives extra weight to the top of the ranking. With `p = 0.5`, the first three positions receive 87.5% of the total weight.",
    "- `jaccard_top3` is stricter because it divides the overlap by the union of both Top-3 sets.",
    "- `overlap_at3` is more forgiving and shows how much of the smaller Top-3 set is recovered by the other set.",
    "",
    "## Interpretation of ranking and top-selection metrics",
    "",
    "### Kendall's tau-b",
    "- Range: approximately -1 to 1",
    "- Positive values indicate that higher-ranked coach options also tend to be higher-ranked model options.",
    "- Values near 0 indicate weak ordering agreement.",
    "",
    "### Tie-adjusted RBO",
    "- Range: 0 to 1",
    "- Higher values indicate stronger similarity, especially near the top of the ranking.",
    "- Ties are handled by splitting positional influence equally across tied items.",
    "",
    "### Jaccard Top 3",
    "- Range: 0 to 1",
    "- `1` means the coach and model Top-3 sets are identical.",
    "- `0` means no overlap at all.",
    "- If there is a tie at rank 3, all tied players are included in the Top-3 set.",
    "",
    "### Overlap@3",
    "- Range: 0 to 1",
    "- Defined here as:",
    "",
    "```",
    "Overlap@3 = |Top3_coach ∩ Top3_model| / min(|Top3_coach|, |Top3_model|)",
    "```",
    "",
    "- `1` means the smaller Top-3 set is fully contained in the other set.",
    "- This is useful when ties at rank 3 expand one or both Top-3 sets beyond three players.",
    "",
    "## Plot files",
    "",
    "### `lmm_full_calibration_plot.png` and `lmm_high_intent_calibration_plot.png`",
    "These plots compare model-predicted coach Scores with observed coach Scores.",
    "",
    "Interpretation:",
    "- Points close to the 45-degree line indicate better calibration.",
    "- A curved smooth line suggests systematic overprediction or underprediction in parts of the score range.",
    "",
    "### `lmm_full_residual_plot.png` and `lmm_high_intent_residual_plot.png`",
    "These plots show fitted values against residuals.",
    "",
    "Interpretation:",
    "- Residuals should be centered around 0 without a strong pattern.",
    "- Curvature suggests non-linearity.",
    "- Increasing spread suggests heteroscedasticity.",
    "",
    "## Supplementary outputs",
    "",
    "### `supplementary_lmm_within_between_full_*` and `supplementary_lmm_within_between_high_intent_*`",
    "These files decompose the selected model score into:",
    "- `model_score_z_within`: within-scene deviation from the scene mean",
    "- `model_score_z_between`: scene-level mean of the model score",
    "",
    "Interpretation:",
    "- The within-scene effect asks whether options that score higher than other options in the same scene also receive higher coach Scores.",
    "- The between-scene effect asks whether scenes with higher average model scores also receive higher average coach Scores.",
    "- For option-evaluation validity, the within-scene effect is usually the more important one.",
    "",
    "## Other files",
    "",
    "### `analysis_settings.csv`",
    "Contains the script settings used for the run.",
    "",
    "### `data_overview.csv`",
    "Contains row and scene counts for the raw dataset, the full analysis dataset, the ICC subset, and the high-intent subset.",
    "",
    "### `scene_size_summary.csv`",
    "Contains descriptive information on scene sizes in the full and high-intent datasets.",
    "",
    "### `full_dataset_row_level_ranks.csv`",
    "Contains row-level recomputed ranks and Top-3 flags for the full dataset.",
    "",
    "### `full_dataset_scene_level_ranking_metrics.csv`",
    "Contains scene-wise tau-b, tie-adjusted RBO, Jaccard Top 3, and Overlap@3 values.",
    "",
    "### `session_info.txt`",
    "Contains the R session information used to generate the outputs.",
    "",
    "## Practical decision guide",
    "",
    "For the main study aim, focus first on:",
    "1. `icc_2_6_results.csv`",
    "2. `lmm_full_fixed_effects.csv`",
    "3. `lmm_full_model_fit.csv`",
    "4. `full_dataset_ranking_metric_summary.csv`",
    "",
    "A convincing validity pattern would usually look like this:",
    "- high ICC for the coach reference",
    "- positive and clearly non-zero mixed-model slope",
    "- meaningful marginal R2 with reasonably low RMSE and MAE",
    "- positive tau-b",
    "- moderate to high top-weighted RBO",
    "- meaningful Top-3 overlap",
    "",
    "If the score-level association is strong but ranking metrics are weak, the model may capture general score level but fail to prioritize the same options as coaches.",
    "If ranking metrics are stronger than absolute-score metrics, the model may be useful for ordering options even if its raw score scale is not well calibrated."
  )

  writeLines(readme_lines, con = file.path(results_dir, "readme.md"))
}

# ----------------------------
# 3) Paths and I/O
# ----------------------------
SCRIPT_DIR <- get_script_dir()
BASE_DIR <- if (dir.exists(SCRIPT_DIR)) SCRIPT_DIR else WINDOWS_BASE_DIR
DATA_PATH <- file.path(BASE_DIR, "coach_ratings.csv")
RESULTS_DIR <- file.path(BASE_DIR, "results")

safe_dir_create(RESULTS_DIR)

if (!file.exists(DATA_PATH)) {
  stop("Could not find `coach_ratings.csv` at: ", DATA_PATH)
}

# ----------------------------
# 4) Load and validate data
# ----------------------------
raw_df <- readr::read_csv(DATA_PATH, show_col_types = FALSE)

required_cols <- c(
  "SceneNr_original", "Scores", "action_intent",
  "Coach1", "Coach2", "Coach3", "Coach4", "Coach5", "Coach6",
  MODEL_SCORE_VAR
)

missing_required_cols <- setdiff(required_cols, names(raw_df))
if (length(missing_required_cols) > 0) {
  stop(
    "The following required columns are missing from the CSV: ",
    paste(missing_required_cols, collapse = ", ")
  )
}

# Main analysis dataset:
# Keep only rows with complete Scores and the selected model score.
full_df <- raw_df %>%
  dplyr::filter(!is.na(Scores), !is.na(.data[[MODEL_SCORE_VAR]])) %>%
  dplyr::mutate(
    SceneNr_original = as.factor(SceneNr_original),
    model_score = .data[[MODEL_SCORE_VAR]]
  )

if (nrow(full_df) == 0) {
  stop("No complete rows remain after filtering for Scores and the selected model score.")
}

# Reference scaling parameters come from the full dataset.
model_score_mean <- mean(full_df$model_score, na.rm = TRUE)
model_score_sd <- stats::sd(full_df$model_score, na.rm = TRUE)

if (!is.finite(model_score_sd) || model_score_sd == 0) {
  stop("The selected model score has zero variance after filtering.")
}

full_df <- full_df %>%
  dplyr::mutate(
    model_score_z = (model_score - model_score_mean) / model_score_sd
  )

high_intent_df <- full_df %>%
  dplyr::filter(action_intent >= HIGH_INTENT_THRESHOLD)

# ICC subset:
coach_cols <- paste0("Coach", 1:6)
icc_df <- full_df %>%
  tidyr::drop_na(dplyr::all_of(coach_cols))

# ----------------------------
# 5) Descriptive outputs
# ----------------------------
analysis_settings <- tibble::tibble(
  setting = c(
    "model_score_var",
    "high_intent_threshold",
    "rbo_p",
    "n_boot",
    "seed",
    "data_path",
    "results_dir",
    "full_model_score_mean",
    "full_model_score_sd"
  ),
  value = c(
    MODEL_SCORE_VAR,
    as.character(HIGH_INTENT_THRESHOLD),
    as.character(RBO_P),
    as.character(N_BOOT),
    as.character(SEED),
    DATA_PATH,
    RESULTS_DIR,
    as.character(model_score_mean),
    as.character(model_score_sd)
  )
)

data_overview <- tibble::tibble(
  dataset = c("raw", "full_complete_scores_and_model", "icc_subset", "high_intent_subset"),
  n_rows = c(nrow(raw_df), nrow(full_df), nrow(icc_df), nrow(high_intent_df)),
  n_scenes = c(
    dplyr::n_distinct(raw_df$SceneNr_original),
    dplyr::n_distinct(full_df$SceneNr_original),
    dplyr::n_distinct(icc_df$SceneNr_original),
    dplyr::n_distinct(high_intent_df$SceneNr_original)
  )
)

scene_size_summary <- dplyr::bind_rows(
  summarise_scene_sizes(full_df, "full"),
  summarise_scene_sizes(high_intent_df, "high_intent")
)

readr::write_csv(analysis_settings, file.path(RESULTS_DIR, "analysis_settings.csv"))
readr::write_csv(data_overview, file.path(RESULTS_DIR, "data_overview.csv"))
readr::write_csv(scene_size_summary, file.path(RESULTS_DIR, "scene_size_summary.csv"))

# ----------------------------
# 6) ICC(2,6)
# ----------------------------
if (nrow(icc_df) > 1) {
  icc_res <- irr::icc(
    icc_df[, coach_cols],
    model = "twoway",
    type = "agreement",
    unit = "average",
    conf.level = 0.95
  )

  icc_output <- tibble::tibble(
    icc_label = "ICC(2,6)",
    description = "Two-way random-effects, absolute agreement, average-measures ICC for 6 raters",
    n_rows = nrow(icc_df),
    n_scenes = dplyr::n_distinct(icc_df$SceneNr_original),
    n_raters = 6,
    icc_value = unname(icc_res$value),
    ci_low_95 = unname(icc_res$lbound),
    ci_high_95 = unname(icc_res$ubound),
    f_value = unname(icc_res$Fvalue),
    df1 = unname(icc_res$df1),
    df2 = unname(icc_res$df2),
    p_value = unname(icc_res$p.value)
  )

  coach_means <- tibble::tibble(
    coach = coach_cols,
    mean_rating = vapply(icc_df[coach_cols], mean, na.rm = TRUE, FUN.VALUE = numeric(1)),
    sd_rating = vapply(icc_df[coach_cols], stats::sd, na.rm = TRUE, FUN.VALUE = numeric(1))
  )

  readr::write_csv(icc_output, file.path(RESULTS_DIR, "icc_2_6_results.csv"))
  readr::write_csv(coach_means, file.path(RESULTS_DIR, "icc_coach_descriptives.csv"))
} else {
  warning("ICC subset has fewer than 2 rows. ICC was not computed.")
}

# ----------------------------
# 7) Main LMMs
# ----------------------------
full_lmm <- fit_main_lmm(full_df, "full", RESULTS_DIR)

if (nrow(high_intent_df) > 1 && dplyr::n_distinct(high_intent_df$SceneNr_original) > 1) {
  high_intent_lmm <- fit_main_lmm(high_intent_df, "high_intent", RESULTS_DIR)
} else {
  warning("High-intent dataset is too small for the mixed model. Skipping high-intent LMM.")
}

# ----------------------------
# 8) Supplementary within/between decomposition
# ----------------------------
supp_full_lmm <- fit_within_between_lmm(full_df, "full", RESULTS_DIR)

if (nrow(high_intent_df) > 1 && dplyr::n_distinct(high_intent_df$SceneNr_original) > 1) {
  supp_high_intent_lmm <- fit_within_between_lmm(high_intent_df, "high_intent", RESULTS_DIR)
}

# ----------------------------
# 9) Ranking and Top-3 validity (full dataset only)
# ----------------------------
full_rank_rows <- make_ranking_output_rows(full_df, MODEL_SCORE_VAR)
readr::write_csv(full_rank_rows, file.path(RESULTS_DIR, "full_dataset_row_level_ranks.csv"))

scene_metrics <- ranking_metrics_by_scene(full_df, MODEL_SCORE_VAR, rbo_p = RBO_P)
readr::write_csv(scene_metrics, file.path(RESULTS_DIR, "full_dataset_scene_level_ranking_metrics.csv"))

ranking_summary <- dplyr::bind_rows(
  bootstrap_metric_summary(scene_metrics$tau_b, "tau_b", n_boot = N_BOOT, seed = SEED),
  bootstrap_metric_summary(scene_metrics$rbo_p_0_5, "rbo_p_0_5", n_boot = N_BOOT, seed = SEED + 1),
  bootstrap_metric_summary(scene_metrics$jaccard_top3, "jaccard_top3", n_boot = N_BOOT, seed = SEED + 2),
  bootstrap_metric_summary(scene_metrics$overlap_at3, "overlap_at3", n_boot = N_BOOT, seed = SEED + 3)
)
readr::write_csv(ranking_summary, file.path(RESULTS_DIR, "full_dataset_ranking_metric_summary.csv"))

# ----------------------------
# 10) Session info and README
# ----------------------------
utils::capture.output(sessionInfo(), file = file.path(RESULTS_DIR, "session_info.txt"))
write_readme(
  results_dir = RESULTS_DIR,
  model_score_var = MODEL_SCORE_VAR,
  high_intent_threshold = HIGH_INTENT_THRESHOLD,
  rbo_p = RBO_P,
  n_boot = N_BOOT
)

message("Analysis complete. Results saved to: ", RESULTS_DIR)
