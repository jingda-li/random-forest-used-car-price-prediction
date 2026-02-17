# ECON 626 — Prediction Competition 4 (PC4)
# Random Forest Regression (R + tidymodels)
#
# Purpose:
#   1) Train a Random Forest model on the provided training set
#   2) Generate predictions for the provided test set
#   3) Export a submission file in the required "4 header lines + predictions" format
#   4) Produce the Q2 diagnostic plot (y vs ŷ on the training set)
#
# Notes:
# - This script keeps the structure of the original submission while:
#     • reducing redundant preprocessing steps
#     • improving comments / readability
#     • writing the submission file using an explicit file connection (open → write → close)

suppressPackageStartupMessages({
  library(tidyverse)
  library(tidymodels)
  library(lubridate)
  library(ranger)
  library(readr)   # for parse_number()
  library(yardstick)
})

# ---------------------------
# File paths / run settings
# ---------------------------
TRAIN_PATH <- "PC3_small_train_data_v1.csv"
TEST_PATH  <- "PC4_test_without_response_variable_v1.csv"

OUT_DIR <- "outputs"
SUBMISSION_PATH <- file.path(OUT_DIR, "pc4.csv")

ANON_NAME <- "Richardo"
STUDENT_ID <- "20902543"

set.seed(626)
if (!dir.exists(OUT_DIR)) dir.create(OUT_DIR, recursive = TRUE)

# ---------------------------
# Load data
# ---------------------------
train_set_raw <- read.csv(TRAIN_PATH)
test_set_raw  <- read.csv(TEST_PATH)

# Response: log(price)
train_set_raw <- mutate(train_set_raw, log_price = log(price))

# ---------------------------
# Preprocessing + feature engineering recipe
# ---------------------------
rec <- recipe(log_price ~ ., data = train_set_raw) %>%
  update_role(price, new_role = "id") %>%

  # Convert mixed-format numeric fields (e.g., "--", "", "34 in") into numeric.
  # parse_number() coerces invalid entries to NA automatically.
  step_mutate(
    listed_date        = ymd(listed_date),
    back_legroom       = parse_number(as.character(back_legroom)),
    front_legroom      = parse_number(as.character(front_legroom)),
    fuel_tank_volume   = parse_number(as.character(fuel_tank_volume)),
    height             = parse_number(as.character(height)),
    length             = parse_number(as.character(length)),
    width              = parse_number(as.character(width)),
    wheelbase          = parse_number(as.character(wheelbase)),
    maximum_seating    = parse_number(as.character(maximum_seating))
  ) %>%

  # Domain-inspired engineered features
  step_mutate(
    age = year(listed_date) - year,
    power_density = if_else(
      !is.na(horsepower) & !is.na(engine_displacement) & engine_displacement > 0,
      horsepower / (engine_displacement / 1000),
      NA_real_
    ),
    avg_mpg = if_else(
      !is.na(city_fuel_economy) & !is.na(highway_fuel_economy),
      (city_fuel_economy + highway_fuel_economy) / 2,
      NA_real_
    ),
    vehicle_footprint = if_else(
      !is.na(length) & !is.na(width),
      length * width,
      NA_real_
    )
  ) %>%

  # Remove columns not used as predictors
  step_rm(
    listed_date, wheel_system_display, year, major_options, engine_cylinders,
    exterior_color, interior_color, city, trim_name, model_name, make_name,
    torque, power
  ) %>%

  # Standardize common missing-value tokens to NA
  step_mutate(
    across(where(is.character), ~na_if(.x, "--")),
    across(where(is.character), ~na_if(.x, ""))
  ) %>%

  # Imputation:
  # - numeric predictors: median
  # - categorical predictors: explicit "Unknown" level for missing values
  step_impute_median(all_numeric_predictors()) %>%
  step_unknown(all_nominal_predictors(), new_level = "Unknown") %>%

  # Ensure nominal predictors are treated as factors before one-hot encoding
  step_string2factor(all_nominal_predictors()) %>%

  # One-hot encode categorical predictors
  step_dummy(all_nominal_predictors(), one_hot = TRUE) %>%

  # Drop predictors with zero variance
  step_zv(all_predictors())

# Bake once to determine p (number of predictors after preprocessing)
prep_rec <- prep(rec, training = train_set_raw)
train_set_baked <- bake(prep_rec, new_data = NULL)
p <- train_set_baked %>% select(-log_price, -price) %>% ncol()

# ---------------------------
# Model specification + workflow
# ---------------------------
rf_spec <- rand_forest(
  mode  = "regression",
  trees = 700,              # B bootstrap samples / trees
  mtry  = floor(sqrt(p)),   # m = sqrt(p)
  min_n = 10                # minimum node size
) %>%
  set_engine("ranger")

wf <- workflow() %>%
  add_recipe(rec) %>%
  add_model(rf_spec)

# Fit on full training data
final_fit <- fit(wf, data = train_set_raw)

# ---------------------------
# Predictions (test set)
# ---------------------------
pred_vals <- predict(final_fit, new_data = test_set_raw)
pred_vals_vec <- pred_vals$.pred

# Training R^2 (computed directly; avoids reliance on engine-specific fields)
train_pred <- predict(final_fit, new_data = train_set_raw)$.pred
R_sqr <- rsq_vec(truth = train_set_raw$log_price, estimate = train_pred)

# ---------------------------
# Export submission file (open → write → close)
# Required format: 4 header lines, then one prediction per line.
# ---------------------------
submission_lines <- c(
  ANON_NAME,
  STUDENT_ID,
  as.character(R_sqr),
  "Random Forest",
  as.character(pred_vals_vec)
)

con <- file(SUBMISSION_PATH, open = "w", encoding = "UTF-8")
on.exit(close(con), add = TRUE)
writeLines(submission_lines, con = con)
message("Wrote submission file to: ", SUBMISSION_PATH)

# ---------------------------
# Q2: y vs ŷ diagnostic plot (training set)
# ---------------------------
png(filename = file.path(OUT_DIR, "q2_y_vs_yhat.png"), width = 900, height = 650)
plot(
  x = train_set_raw$log_price,
  y = train_pred,
  xlab = "y (log price)",
  ylab = "ŷ (predicted log price)",
  main = "Training set: y vs ŷ",
  pch  = 16,
  cex  = 0.25
)
abline(a = 0, b = 1, col = "red", lwd = 3)
dev.off()
message("Saved Q2 plot to: ", file.path(OUT_DIR, "q2_y_vs_yhat.png"))
