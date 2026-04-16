# ------------------------------
# Data preparation
# ------------------------------

setwd("C:\\252\\MT2013\\BTL")

# Set the download server
options(repos = c(CRAN = "https://cran.r-project.org")) 

if (!require("pacman")) install.packages("pacman")
library(pacman)

# Load packages
pacman::p_load(caret, dplyr, e1071, ggcorrplot, ggplot2,
               glmnet, nnls, patchwork, psych, 
               randomForest, ranger, tibble, tidyverse, 
               tidyr, xgboost, yardstick)

data <- read.csv("data.csv")

# Convert text categories into binary numbers
temp_data <- data |>
  dplyr::mutate(infill_pattern = as.numeric(infill_pattern != "grid"),
                material = as.numeric(material != "abs"))

print(summary(temp_data), justify = "left")

# Reshape using modern pivot_longer (replacing deprecated gather)
gathered_data <- temp_data |> 
  pivot_longer(cols = everything(), names_to = "key", values_to = "value") |>
  mutate(key = factor(key, levels = unique(colnames(temp_data))))

# ------------------------------
# Chap 3: Data Visualization
# ------------------------------

# ---------- Histogram ----------
# Lock in the order of the variables so the plots stay organized
gathered_data <- gathered_data |>
  mutate(key = factor(key, levels = unique(key)))

# Plot a grid of histograms to show the distribution of each variable
(ggplot(data = gathered_data, aes(x = value)) + 
    geom_histogram(color = "black", fill = "lightblue", bins = 30) +
    labs(x = "Value", y = "Frequency", title = "Histogram") +
    facet_wrap(~ key, scales = "free", nrow = 4)) |>
  print()

# ---------- Box plot ----------
# Remove categorical variables since boxplots are for continuous numbers
# Plot a grid of boxplots to easily spot outliers and data spread
gathered_data |>
  filter(key != "infill_pattern" & key != "material") |>
  ggplot(aes(y = as.numeric(value))) +
  stat_boxplot(geom = "errorbar", width = 0.2) +
  geom_boxplot(outlier.shape = NA, fill = "lightblue") +
  labs(y = "Value", title = "Box plot") +
  facet_wrap(~ key, scales = "free", nrow = 3) |>
  print()

# ---------- Correlation matrix ----------
# Calculate how strongly variables relate to each other, rounded to 2 decimals
cor_mat_data <- round(cor(temp_data), 2)

# Visualize these relationships as a color-coded heatmap grid
ggcorrplot(cor_mat_data, method = "square", type = "upper", lab = TRUE,
           title = "Correlation Matrix", 
           legend.title = "Pearson\nCorrelation\n") |>
  print()

# ---------- Scatter plots ----------

# p1: Compare all predictor variables against 'roughness'
p1 <- temp_data |>
  select(-c(infill_pattern, tension_strenght, elongation)) |> 
  pivot_longer(cols = -c(roughness, material), 
               names_to = "key", 
               values_to = "value") |> 
  ggplot(aes(x = value, y = roughness, color = factor(material))) +
  geom_point(size = 3, alpha = 0.7) +
  labs(x = "Values", y = "Roughness", 
       title = "Variables against Roughness", color = "Material Type") +
  facet_wrap(~ key, scales = "free", nrow = 3) +
  scale_color_manual(values = c("0" = "tomato", "1" = "aquamarine3"),
                     labels = c("ABS", "PLA"))

# p2: Compare all predictor variables against 'tension_strenght'
p2 <- temp_data |>
  select(-c(infill_pattern, roughness, elongation)) |> 
  pivot_longer(cols = -c(tension_strenght, material), 
               names_to = "key", 
               values_to = "value") |> 
  ggplot(aes(x = value, y = tension_strenght, color = factor(material))) +
  geom_point(size = 3, alpha = 0.7) +
  labs(x = "Values", y = "Tension strenght", 
       title = "Variables against Tension strenght", color = "Material Type") +
  facet_wrap(~ key, scales = "free", nrow = 3) +
  scale_color_manual(values = c("0" = "tomato", "1" = "aquamarine3"),
                     labels = c("ABS", "PLA"))

# p3: Compare all predictor variables against 'elongation'
p3 <- temp_data |>
  select(-c(infill_pattern, tension_strenght, roughness)) |> 
  pivot_longer(cols = -c(elongation, material), 
               names_to = "key", 
               values_to = "value") |> 
  ggplot(aes(x = value, y = elongation, color = factor(material))) +
  geom_point(size = 3, alpha = 0.7) +
  labs(x = "Values", y = "Elongation", 
       title = "Variables against Elongation", color = "Material Type") +
  facet_wrap(~ key, scales = "free", nrow = 3) +
  scale_color_manual(values = c("0" = "tomato", "1" = "aquamarine3"),
                     labels = c("ABS", "PLA"))

# Print the scatter plots to the viewing window
print(p1)
print(p2)
print(p3)


# ------------------------------
# Chap 4: Inferential Statistics & Modeling Setup
# ------------------------------

outputs <- c("roughness", "tension_strenght", "elongation")
inputs <- c("layer_height", "wall_thickness", "nozzle_temperature", 
            "bed_temperature", "print_speed", "fan_speed", "infill_density",
            "infill_pattern", "material")

# ---------- Correlation tests ----------
for (y in outputs) {
  print(paste("===== Output:", y, "====="))
  for (x in inputs) {
    test <- cor.test(temp_data[[x]], temp_data[[y]])
    cat(sprintf("Input: %-18s: Correlation: %-10.4f p_value: %-10.2e\n", 
                x, test$estimate, test$p.value))
  }
}

# ---------- ANOVA ----------
for (y in outputs) {
  model <- aov(formula(paste(y, "~ material + infill_pattern")), data=temp_data)
  print(summary(model), justify = "left")
}

# ---------- Data Encoding & Split ----------
temp_data <- temp_data |>
  mutate(material_f = factor(material),
         infill_pattern_f = factor(infill_pattern),
         material_num = as.numeric(material_f) - 1,
         infill_pattern_num = as.numeric(infill_pattern_f) - 1)

inputs_num <- c("layer_height","wall_thickness","nozzle_temperature","bed_temperature",
                "print_speed","fan_speed","infill_density","infill_pattern_num","material_num")

inputs_tree <- c("layer_height","wall_thickness","nozzle_temperature","bed_temperature",
                 "print_speed","fan_speed","infill_density","infill_pattern_f","material_f")

set.seed(42) # Ensure reproducibility for splitting
temp_data$.strata <- ifelse(temp_data[[outputs[1]]] > median(temp_data[[outputs[1]]]), "high", "low")
train_idx <- createDataPartition(temp_data$.strata, p = 0.8, list = FALSE)
train_df <- temp_data[train_idx, ]
test_df  <- temp_data[-train_idx, ]
n_train <- nrow(train_df)

# ---------- Base model training functions ----------

train_enet <- function(x_train, y_train, x_val, y_val, preproc = NULL){
  if(!is.null(preproc)){
    x_train <- predict(preproc, x_train)
    x_val   <- predict(preproc, x_val)
  }
  
  best_rmse <- Inf; best_model <- NULL; best_alpha <- NULL
  set.seed(42)
  for(a in seq(0, 1, by = 0.25)){ 
    cv <- cv.glmnet(as.matrix(x_train), y_train, alpha = a, nfolds = 5)
    preds <- as.numeric(predict(cv, as.matrix(x_val), s = "lambda.min"))
    rmse <- sqrt(mean((y_val - preds)^2))
    
    if(rmse < best_rmse){
      best_rmse <- rmse
      best_model <- cv
      best_alpha <- a
    }
  }
  preds_final <- as.numeric(predict(best_model, as.matrix(x_val), s = "lambda.min"))
  list(model = best_model, preds = preds_final, alpha = best_alpha)
}

train_rf <- function(df_train, y_train, df_val, y_val){
  best_rmse <- Inf; best_model <- NULL; best_params <- list()
  set.seed(42) 
  for(m in c(2,4,6,8)){                     
    for(n in c(1,3,5,10,20)){               
      model <- ranger::ranger(
        y ~ ., data = data.frame(y=y_train, df_train),
        mtry = m, min.node.size = n, num.trees = 500, 
        sample.fraction = 0.8, replace = TRUE,
        importance = "impurity", seed = 42 
      )
      preds <- predict(model, data = df_val)$predictions
      rmse <- sqrt(mean((y_val - preds)^2))
      
      if(rmse < best_rmse){
        best_rmse <- rmse
        best_model <- model
        best_params <- list(mtry = m, min.node.size = n)
      }
    }
  }
  preds_final <- predict(best_model, data = df_val)$predictions
  list(model = best_model, preds = preds_final, params = best_params)
}

train_xgb <- function(x_train, y_train, x_val, y_val){
  dtrain <- xgboost::xgb.DMatrix(as.matrix(x_train), label = y_train)
  dval   <- xgboost::xgb.DMatrix(as.matrix(x_val), label = y_val)
  
  best_rmse <- Inf; best_model <- NULL; best_params <- list()
  set.seed(42) 
  for(eta in c(0.01, 0.05, 0.1)){
    for(depth in c(3,4,6)){
      model <- xgboost::xgb.train(
        params = list(objective = "reg:squarederror", eta = eta, max_depth = depth,
                      subsample = 0.8, colsample_bytree = 0.8, seed = 42), 
        data = dtrain, nrounds = 1000, watchlist = list(val = dval),
        early_stopping_rounds = 20, verbose = 0
      )
      preds <- predict(model, as.matrix(x_val))
      rmse <- sqrt(mean((y_val - preds)^2))
      
      if(rmse < best_rmse){
        best_rmse <- rmse
        best_model <- model
        best_params <- list(eta = eta, max_depth = depth, best_ntreelimit = model$best_iteration)
      }
    }
  }
  preds_final <- predict(best_model, as.matrix(x_val))
  list(model = best_model, preds = preds_final, params = best_params)
}

# ---------- OOF predictions & Weight Calculation ----------
K <- 5 
folds <- createFolds(train_df[[outputs[1]]], k = K, list = TRUE, returnTrain = FALSE)
model_names <- c("enet","rf","xgb")

oof_preds <- list()
oof_rmse <- data.frame(Output = outputs, OOF_RMSE = NA)

for (out in outputs) {
  cat("Generating OOF for", out, "\n")
  A <- matrix(NA, nrow = n_train, ncol = length(model_names))
  colnames(A) <- model_names
  
  for (k in seq_along(folds)) {
    val_idx <- folds[[k]]
    train_idx_fold <- setdiff(seq_len(n_train), val_idx)
    
    df_tr <- train_df[train_idx_fold, ]
    df_val<- train_df[val_idx, ]
    y_val <- df_val[[out]]
    
    preproc_obj <- preProcess(df_tr[, inputs_num], method = c("center","scale"))
    
    r_enet <- train_enet(df_tr[, inputs_num], df_tr[[out]], df_val[, inputs_num], y_val, preproc = preproc_obj)
    rf_res <- train_rf(df_tr[, inputs_tree], df_tr[[out]], df_val[, inputs_tree], y_val)
    xgb_res <- train_xgb(predict(preproc_obj, df_tr[, inputs_num]), df_tr[[out]], predict(preproc_obj, df_val[, inputs_num]), y_val)
    
    A[val_idx, "enet"] <- r_enet$preds
    A[val_idx, "rf"]   <- rf_res$preds
    A[val_idx, "xgb"]  <- xgb_res$preds
  }
  
  oof_rmse_val <- sqrt(mean((train_df[[out]] - rowMeans(A, na.rm=TRUE))^2))
  oof_rmse$OOF_RMSE[oof_rmse$Output==out] <- oof_rmse_val
  oof_preds[[out]] <- A
}

print(oof_rmse)

# Compute NNLS weights
weights <- list()
for (out in outputs){
  nn <- nnls::nnls(oof_preds[[out]], train_df[[out]])
  w_raw <- coef(nn)
  w_norm <- if(sum(w_raw) > 0) w_raw / sum(w_raw) else w_raw
  weights[[out]] <- list(raw = w_raw, norm = w_norm)
}

cat("\n=========================================\n")
cat("       NNLS NORMALIZED WEIGHTS (%)\n")
cat("=========================================\n")

for (out in outputs) {
  cat(sprintf("Target: %-18s\n", toupper(out)))
  
  w_norm <- weights[[out]]$norm
  for (i in seq_along(model_names)) {
    cat(sprintf("  %-5s : %6.2f %%\n", toupper(model_names[i]), w_norm[i] * 100))
  }
  cat("-----------------------------------------\n")
}


# ---------- Train final base models ----------
final_models <- list()
preds_test_mat <- list()

for (out in outputs) {
  cat("\nTraining Final Models for", out, "...\n")
  y_tr <- train_df[[out]]
  
  preproc_full <- preProcess(train_df[, inputs_num], method = c("center","scale"))
  
  res_enet <- train_enet(train_df[, inputs_num], y_tr, train_df[, inputs_num], y_tr, preproc = preproc_full)
  res_rf   <- train_rf(train_df[, inputs_tree], y_tr, train_df[, inputs_tree], y_tr)
  res_xgb  <- train_xgb(predict(preproc_full, train_df[, inputs_num]), y_tr, 
                        predict(preproc_full, train_df[, inputs_num]), y_tr)
  
  final_models[[out]] <- list(preproc = preproc_full, enet = res_enet$model, rf = res_rf$model, xgb = res_xgb$model)
  
  enet_p <- as.numeric(predict(res_enet$model, as.matrix(predict(preproc_full, test_df[, inputs_num])), s = "lambda.min"))
  rf_p   <- predict(res_rf$model, data = test_df[, inputs_tree])$predictions
  xgb_p  <- predict(res_xgb$model, as.matrix(predict(preproc_full, test_df[, inputs_num])))
  
  preds_test_mat[[out]] <- cbind(enet = enet_p, rf = rf_p, xgb = xgb_p)
}

# ==============================
# Evaluation & Preparing NNLS Preds
# ==============================

calc_metrics <- function(y_true, y_pred){
  rmse <- sqrt(mean((y_true - y_pred)^2)) 
  mae  <- mean(abs(y_true - y_pred))      
  r2   <- 1 - sum((y_true - y_pred)^2) / sum((y_true - mean(y_true))^2) 
  return(c(RMSE=rmse, MAE=mae, R2=r2))
}

eval_tbl <- tibble(Output = character(), RMSE = numeric(), MAE = numeric(), R2 = numeric())

for (out in outputs){
  # Extract the NNLS weights and the test predictions matrix
  w <- weights[[out]]$norm
  preds_test <- preds_test_mat[[out]]
  
  # Calculate the final blended prediction using matrix multiplication
  blended_test <- as.numeric(preds_test %*% w)
  
  # Update the main matrix with the new NNLS blend column
  preds_test_mat[[out]] <- cbind(preds_test, nnls_blend = blended_test)
  
  y_true <- test_df[[out]]
  metrics <- calc_metrics(y_true, blended_test)
  
  eval_tbl <- bind_rows(eval_tbl,
                        tibble(Output = out,
                               RMSE = as.numeric(metrics["RMSE"]),
                               MAE  = as.numeric(metrics["MAE"]),
                               R2   = as.numeric(metrics["R2"])))
  
  # ==========================================================
  # PHẦN 1: IN KẾT QUẢ 3 MÔ HÌNH CƠ SỞ (BASE MODELS)
  # ==========================================================
  cat("\n\n")
  cat("==========================================================\n")
  cat(sprintf("       BASE MODELS PERFORMANCE FOR: %s\n", toupper(out)))
  cat("==========================================================\n")
  
  # Chỉ lặp qua enet, rf, xgb
  base_models <- c("enet", "rf", "xgb")
  for (model_name in base_models){
    m <- calc_metrics(y_true, preds_test_mat[[out]][, model_name])
    cat(sprintf("%-10s: RMSE = %-7.4f | MAE = %-7.4f | R2 = %-7.4f\n", 
                toupper(model_name), round(m["RMSE"],4), round(m["MAE"],4), round(m["R2"],4)))
  }
  
  # ==========================================================
  # PHẦN 2: IN KẾT QUẢ MÔ HÌNH KẾT HỢP (NNLS ENSEMBLE)
  # ==========================================================
  cat("\n----------------------------------------------------------\n")
  cat(sprintf("         NNLS ENSEMBLE PERFORMANCE FOR: %s\n", toupper(out)))
  cat("----------------------------------------------------------\n")
  
  # Tính toán và in riêng cho NNLS
  m_nnls <- calc_metrics(y_true, preds_test_mat[[out]][, "nnls_blend"])
  cat(sprintf("%-10s: RMSE = %-7.4f | MAE = %-7.4f | R2 = %-7.4f\n", 
              "NNLS_BLEND", round(m_nnls["RMSE"],4), round(m_nnls["MAE"],4), round(m_nnls["R2"],4)))
}

print(eval_tbl)

# ==============================
# Visualization for ML Models & NNLS
# ==============================

# 1. Plot tỷ trọng (Weights) của NNLS
weights_df <- data.frame()
for(out in outputs) {
  weights_df <- rbind(weights_df, 
                      data.frame(Output = out, 
                                 Model = model_names, 
                                 Weight = weights[[out]]$norm))
}

p_weights <- ggplot(weights_df, aes(x = Output, y = Weight, fill = Model)) +
  geom_bar(stat = "identity", position = "dodge", color = "black", alpha = 0.8) +
  scale_y_continuous(labels = scales::percent_format()) +
  scale_fill_manual(values = c("enet" = "#E69F00", "rf" = "#56B4E9", "xgb" = "#009E73")) +
  labs(title = "NNLS Ensemble Weights per Target", x = "Target Variable", y = "Weight (%)") +
  theme_minimal(base_size = 13) +
  theme(plot.title = element_text(face = "bold", hjust = 0.5),
        legend.position = "bottom")
print(p_weights)

# 2. Plot Actual vs Predicted 
plot_actual_vs_pred <- function(y_true, y_pred, title){
  df <- data.frame(actual = y_true, pred = y_pred)
  ggplot(df, aes(x = actual, y = pred)) +
    geom_jitter(width = 0.08, height = 0.08, size = 2.5, alpha = 0.5, color = "#2E86C1") +
    geom_abline(slope = 1, intercept = 0, color = "red", linetype = "dashed", linewidth = 1) + 
    labs(title = title, x = "Actual", y = "Predicted") +
    theme_minimal(base_size = 13) +
    theme(plot.title = element_text(face = "bold", hjust = 0.5),
          panel.grid.minor = element_blank())
}

for (out in outputs){
  y_true <- test_df[[out]]
  preds <- preds_test_mat[[out]]
  
  p1_model <- plot_actual_vs_pred(y_true, preds[, "enet"], paste("ENET -", out))
  p2_model <- plot_actual_vs_pred(y_true, preds[, "rf"], paste("RF -", out))
  p3_model <- plot_actual_vs_pred(y_true, preds[, "xgb"], paste("XGB -", out))
  p4_model <- plot_actual_vs_pred(y_true, preds[, "nnls_blend"], paste("NNLS ENSEMBLE -", out)) +
    theme(plot.title = element_text(color = "darkred")) 
  
  combined_plot <- (p1_model | p2_model) / (p3_model | p4_model) + 
    plot_annotation(title = paste("Model Performance Comparison:", out),
                    theme = theme(plot.title = element_text(size = 16, face = "bold", hjust = 0.5)))
  
  print(combined_plot)
}

write.csv(eval_tbl, "evaluation_test_nnls.csv", row.names = FALSE)
saveRDS(weights, "nnls_weights.rds")