# PROJECT DESCRIPTION
# 
# Compare the performance of various levels of cost sensitivity in the logistic regression
# applied to a credit card fraud dataset.
# 
# The project includes the implementation of several methods in the pre-processing, processing, 
# and post-processing phases of the fradu detection model development
#

rm(list = ls())

library(devtools)
library(caTools)
library(cslogit)
library(ggplot2)
library(xgboost)
library(gridExtra)
library(lubridate)
library(splitstackshape)
library(dplyr)
library(PerformanceMetrics)
library(tidyverse)
library(ROCR) 
library(DescTools)
library(timeSeries)
library(caret)


### Load data
load("creditcard_complete.RData")

### Adjustble parameters
nruns  <- 5 
nfolds <- 2
set.seed(2019)
resample_setting <- "RAW" #Options for resample_setting: "ROS", "SMOTE", "CS-ROS", "RUS", "CSRS_ID","CSRS_CD", "RAW"
oversample_size <- 10.0


### Pre-processing 
creditcard <- creditcard[-which(creditcard$Amount == 0), ] # remove transactions with zero Amount
creditcard <- creditcard[, -1] # remove variable Time
creditcard$Class <- factor(creditcard$Class) # set Class to factor variable
rownames(creditcard) <- 1:nrow(creditcard)
amount <- creditcard$Amount
creditcard$LogAmount <- log(creditcard$Amount)# log-transformation of Amount
creditcard <- creditcard[, c(1:28, 31, 30)] # rearrange columns
creditcard[, -30] <- scale(creditcard[, -30])


### Amount category, used for stratification:
print(quantile(amount[creditcard$Class == 1], probs = c(1/3, 2/3)))
amount_category <- rep("high", length(amount))
amount_category[amount < quantile(amount[creditcard$Class == 1], probs = 2/3)] <- "middle"
amount_category[amount < quantile(amount[creditcard$Class == 1], probs = 1/3)] <- "low"
amount_category <- factor(amount_category, levels = c("low", "middle", "high"))

print(table(amount_category))
print(prop.table(table(amount_category)))
print(table(Class = creditcard$Class, amount_category = amount_category))


### Create folds (nruns x nfolds) for cross-validation 
cvfolds <- list()
for (k in 1:nruns) {
  folds <- stratified(indt = cbind.data.frame(Class = creditcard$Class,
                                              amount_category = amount_category),
                      group = c("Class", "amount_category"),
                      size  = 0.5, # 2-fold cross validation (= 1/nfolds)
                      bothSets = TRUE,
                      keep.rownames = TRUE)
  names(folds) <- c("Fold1", "Fold2")
  folds[[1]] <- sort(as.numeric(folds[[1]]$rn))
  folds[[2]] <- sort(as.numeric(folds[[2]]$rn))
  cvfolds[[k]] <- folds
}


### Train the models
coefficients_logit <- c()
coefficients_class <- c()
coefficients_instance <- c()

number_instance_prior <- c()
number_instance_post <- c()
logit_AIC_R2 <-c() 

fitted_values_logit <- c()
fitted_values_class <- c()
fitted_values_instance <- c()

prediction_logit <- c()
prediciton_class <- c()
prediciton_instance <- c()

results_logit_threshold_0.5_train <- c()
results_class_logit_threshold_0.5_train <- c()
results_instance_logit_threshold_0.5_train <- c()

results_logit_threshold_0.5 <- c()
results_class_logit_threshold_0.5 <- c()
results_instance_logit_threshold_0.5 <- c()
results_logit_threshold_imbalance_ratio <- c()
results_class_logit_threshold_imbalance_ratio <- c()
results_instance_logit_threshold_imbalance_ratio <- c()
results_logit_threshold_class_cost <- c()
results_class_logit_threshold_class_cost <- c()
results_instance_logit_threshold_class_cost <- c()
results_logit_threshold_instance_cost <- c()
results_class_logit_threshold_instance_cost <- c()
results_instance_logit_threshold_instance_cost <- c()
results_logit_threshold_Youden_optimal <- c()
results_class_logit_threshold_Youden_optimal <- c()
results_instance_logit_threshold_Youden_optimal <- c()
results_logit_threshold_F1_optimal <- c()
results_class_logit_threshold_F1_optimal <- c()
results_instance_logit_threshold_F1_optimal <- c()
results_logit_threshold_Savings_optimal <- c()
results_class_logit_threshold_Savings_optimal <- c()
results_instance_logit_threshold_Savings_optimal <- c()

testset_with_prediction_logit <- c()
testset_with_prediction_class <- c()
testset_with_prediction_instance <- c()

thresholds_constant <- c()
threshold_BMR <- c()

amount_test_cv <- c()

time_logit   <- c()
time_instance_logit <- c()
time_class_logit <- c()
t_start <- proc.time()

for (k_run in 1:nruns) {  
  for (j_fold in 1:nfolds) {
    
    cat(paste0("\n Run ", k_run, "/", nruns, " - fold ", j_fold, "/", nfolds, "\n"))
    
    
    ### PRE-PROCESSING-----------------------------------------------------------
   
    
    ## Create train-test split
    # Cost-insensitive
    train <- creditcard[-cvfolds[[k_run]][[j_fold]], ] 
    test <- creditcard[ cvfolds[[k_run]][[j_fold]], ]
    # Amounts; needed later
    amount_train <- amount[-cvfolds[[k_run]][[j_fold]]]
    amount_test <- amount[ cvfolds[[k_run]][[j_fold]]]
    fixed_cost <- 10
    avg_amount_train <- mean(amount_train)
    avg_amount_test <- avg_amount_train
    
    
    ## Amount category, used for stratification (test-validation split):
    amount_category <- rep("high", length(amount_test))
    amount_category[amount_test < quantile(amount[test$Class == 1], probs = 2/3)] <- "middle"
    amount_category[amount_test < quantile(amount[test$Class == 1], probs = 1/3)] <- "low"
    amount_category <- factor(amount_category, levels = c("low", "middle", "high"))
  
    
    ## Create test and validation sets
    folds <- stratified(indt = cbind.data.frame(Class = test$Class,
                                                amount_category = amount_category),
                        group = c("Class", "amount_category"),
                        size  = 0.5, # 2-fold cross validation (= 1/nfolds)
                        bothSets = TRUE,
                        keep.rownames = TRUE)
    names(folds) <- c("test", "validation")
    folds[[1]] <- sort(as.numeric(folds[[1]]$rn))
    folds[[2]] <- sort(as.numeric(folds[[2]]$rn))
    
    validation <- test[folds[[2]], ]
    test <- test[folds[[1]], ] 
  
    amount_validation <- amount_test[folds[[2]]]
    amount_test <- amount_test[folds[[1]]] 
    
    
    ## Instances
    number_instance_prior <- rbind.data.frame(number_instance_prior, cbind.data.frame(k_run, j_fold, length(train$Class), 
                                                                                      sum(train$Class ==1), length(test$Class),
                                                                                      sum(test$Class ==1), length(validation$Class), 
                                                                                      sum(validation$Class ==1), length(amount_train), 
                                                                                      length(amount_test), length(amount_validation)))
    
    
    ## Resampling
    if(resample_setting == "ROS") {
      # ROS
      oversampled_cs_var1_object <- ROSE::ovun.sample(Class ~ ., train, "over", p = 0.5)
      train <- oversampled_cs_var1_object$data
    }
    else if(resample_setting == "CS-ROS") {
      # BAHNSEN
      full_amount_train <- train
      full_amount_train$Amount <- amount_train
      N_pos <- full_amount_train[-which(full_amount_train$Class == 0), ]
      min_cost <- abs(min(N_pos$Amount))
      max_cost <- quantile(N_pos$Amount, 0.85)
      weight_matrix <- matrix(nrow = nrow(train), ncol = 1)
      weight_matrix[, 1] <- ifelse(train$Class == 1, floor(oversample_size*(full_amount_train$Amount)/max_cost), 1) # FN, FP
      # Oversample
      idx <- rep(1:nrow(train), weight_matrix[, 1])
      train <- train[idx,]
    }
    
    else if (resample_setting == "RUS") {
      # Random undersampling (library(caret))
      train <- downSample(x = train[ ,1:29], y = train$Class) # Random undersampling for training set
    }
    
    else if (resample_setting == "CSRS_ID") {
      # Cost sensitive reject sampling - instance dependent (Bahsen et al)
      cost_mat <- matrix(nrow = nrow(train), ncol = 2) # false positives, false negatives
      cost_miss <- c(nrow = nrow(train))
      wc <- c(nrow = nrow(train))
      max_wc <- 0.975
      filter_rej <- c(nrow = nrow(train))
      cost_mat[, 1] <- fixed_cost # FP
      cost_mat[, 2] <- amount_train # FN
      cost_miss <- ifelse(train$Class == 1, cost_mat[, 2], cost_mat[, 1])
      cost_miss_w <- cost_miss/quantile(cost_miss, probs = max_wc) # ALTERNATIVES cost_miss_w <- cost_miss/max(cost_miss) OR cost_miss_w <- cost_miss/ (max(cost_miss)+0.1)
      wc <- ifelse(cost_miss_w <1, cost_miss_w, 1)
      rej_rand <- runif(NROW(train), 0, 1)
      filter_rej <- rej_rand <= wc
      train <- subset(x=train, filter_rej == TRUE) # Cost sensitive reject sampling for instance dependent sample
    }
    
    else if (resample_setting == "RAW") {
      train <-train
    }
    
    
    ## Amount train
    amount_train <- amount[as.integer(rownames(train))]
    
    
    ## Create cost matrices  
    # Cost Insensitive train   
    insensitive_matrix_train <- matrix(nrow = nrow(train), ncol = 2)
    insensitive_matrix_train[, 1] <- ifelse(train$Class == 1, 0, 0) # TP, TN
    insensitive_matrix_train[, 2] <- ifelse(train$Class == 1, 1, 1) # FN, FP
    # Class dependent train
    avg_amount <- mean(amount_train)
    fixed_cost <- 10
    class_matrix_train <- matrix(nrow = nrow(train), ncol = 2)
    class_matrix_train[, 1] <- ifelse(train$Class == 1, fixed_cost, 0) # TP, TN
    class_matrix_train[, 2] <- ifelse(train$Class == 1, avg_amount, fixed_cost) # FN, FP
    # Instance dependent train 
    fixed_cost <- 10
    instance_matrix_train <- matrix(nrow = nrow(train), ncol = 2)
    instance_matrix_train[, 1] <- ifelse(train$Class == 1, fixed_cost, 0) # TP, TN
    instance_matrix_train[, 2] <- ifelse(train$Class == 1, amount_train, fixed_cost) # FN, FP
    # Instance dependent test    
    fixed_cost <- 10
    instance_matrix_test <- matrix(nrow = nrow(test), ncol = 2)
    instance_matrix_test[, 1] <- ifelse(test$Class == 1, fixed_cost, 0) # TP, TN
    instance_matrix_test[, 2] <- ifelse(test$Class == 1, amount_test, fixed_cost) # FN, FP
    # Instance dependent validation (needed in findOptimalThreshold(Savings))
    fixed_cost <- 10
    instance_matrix_validation <- matrix(nrow = nrow(validation), ncol = 2)
    instance_matrix_validation[, 1] <- ifelse(validation$Class == 1, fixed_cost, 0) # TP, TN
    instance_matrix_validation[, 2] <- ifelse(validation$Class == 1, amount_validation, fixed_cost) # FN, FP
    
    
    ## Remove instances where amount is lower than fixed_cost
    low_amount_train <- which(amount_train < fixed_cost)
    train <- train[-low_amount_train, ]
    insensitive_matrix_train <- insensitive_matrix_train[-low_amount_train, ]
    class_matrix_train <- class_matrix_train[-low_amount_train, ]
    instance_matrix_train <- instance_matrix_train[-low_amount_train, ]
    amount_train <- amount_train[-low_amount_train]
    
    
    ## Instance count
    number_instance_post <- rbind.data.frame(number_instance_post, cbind.data.frame(k_run, j_fold, length(train$Class),
                                                                                    sum(train$Class ==1), length(test$Class), 
                                                                                    sum(test$Class ==1), length(validation$Class), 
                                                                                    sum(validation$Class ==1), length(amount_train), 
                                                                                    length(amount_test), length(amount_validation)))
    
    
    ### PROCESSING--------------------------------------------------------------
    
    
    ## Cost-insensitive       
    t_start_logit <- proc.time()
    logit <- glm(formula = Class ~ ., data = train, family = "binomial")
    t_end_logit <- proc.time() - t_start_logit
    time_logit <- c(time_logit, t_end_logit[3])
    
    
    ## Class dependent   
    class_logitL1 <- cslogit(formula     = Class ~ .,
                             data        = train,
                             cost_matrix = class_matrix_train,
                             lambda      = 0,
                             options     = list(check_data = FALSE,
                                                start = logit$coefficients))
    time_class_logit <- c(time_class_logit, class_logitL1$time)
    
    
    ## Instance dependent
    instance_logitL1 <- cslogit(formula     = Class ~ .,
                                data        = train,
                                cost_matrix = instance_matrix_train,
                                lambda      = 0,
                                options     = list(check_data = FALSE,
                                                   start = logit$coefficients))
    time_instance_logit <- c(time_instance_logit, instance_logitL1$time)
  
    
    ## Coefficients
    coefficients_logit <- rbind.data.frame(coefficients_logit, cbind.data.frame(k_run, j_fold, t(logit$coefficients)))
    coefficients_class <- rbind.data.frame(coefficients_class, cbind.data.frame(k_run, j_fold, t(class_logitL1$coefficients)))
    coefficients_instance <- rbind.data.frame(coefficients_instance, cbind.data.frame(k_run, j_fold, t(instance_logitL1$coefficients)))
    
    
    ## Fitted values
    fitted_values_logit_run <- predict(logit, newdata = train, type="response")
    fitted_values_logit <- rbind.data.frame(fitted_values_logit, cbind.data.frame(k_run, j_fold, (fitted_values_logit_run),train$Class))
    fitted_values_class <- rbind.data.frame(fitted_values_class, cbind.data.frame(k_run, j_fold, (class_logitL1$fitted_values), train$Class))
    fitted_values_instance <- rbind.data.frame(fitted_values_instance, cbind.data.frame(k_run, j_fold, (instance_logitL1$fitted_values),train$Class))
    
    
    ## Performance function
    performanceMeasures <- function (scores, threshold, true_classes, cost_matrix) {
      predicted_classes <- ifelse(scores > threshold, 1, 0)
      metrics <- PerformanceMetrics::performance(scores, predicted_classes, true_classes, cost_matrix, plot = FALSE)$metrics
      return(metrics)
    }
    
    
    ## Scores on test data
    scores_logit_test   <- predict(logit, newdata = test, type="response")
    scores_class_logit_test <- predict(class_logitL1, newdata = test) 
    scores_instance_logit_test <- predict(instance_logitL1, newdata = test) 
    
    
    ## Scores on validation data
    scores_logit_validation   <- predict(logit, newdata = validation, type="response")
    scores_class_logit_validation <- predict(class_logitL1, newdata = validation) 
    scores_instance_logit_validation <- predict(instance_logitL1, newdata = validation) 
    
    
    if( nruns==5 ){
      ## Merge test set with prediction
      testset_with_prediction_logit <- rbind.data.frame(testset_with_prediction_logit, cbind.data.frame(k_run, j_fold, test, scores_logit_test))
      testset_with_prediction_class <- rbind.data.frame(testset_with_prediction_class, cbind.data.frame(k_run, j_fold, test, scores_class_logit_test))
      testset_with_prediction_instance <- rbind.data.frame(testset_with_prediction_instance, cbind.data.frame(k_run, j_fold, test, scores_instance_logit_test))
      
      
      ## Amount test set
      amount_test_cv <- rbind.data.frame(amount_test_cv, cbind.data.frame(k_run, j_fold, amount_test))
        
      
      ### POST-PROCESSING----------------------------------------------------------
    
      
      ## Thresholds
      threshold_0.5 <- 0.5
      threshold_imbalance_ratio <- sum(validation$Class=="1")/nrow(validation)
      threshold_class_cost <- fixed_cost / mean(amount_validation) 
      threshold_instance_cost <- fixed_cost / amount_test 
    
      threshold_Youden_optimal_logit <- findOptimalThreshold(scores = scores_logit_validation, true_classes = validation$Class, thresholds=c(seq(0.001, 0.999, 0.002)), metric = "Youden", plot = F)
      threshold_Youden_optimal_logit <- threshold_Youden_optimal_logit$optimal_threshold
    
      threshold_Youden_optimal_class <- findOptimalThreshold(scores = scores_class_logit_validation, true_classes = validation$Class, thresholds=c(seq(0.001, 0.999, 0.002)), metric = "Youden", plot = F)
      threshold_Youden_optimal_class <- threshold_Youden_optimal_class$optimal_threshold
    
      threshold_Youden_optimal_instance <- findOptimalThreshold(scores = scores_instance_logit_validation, true_classes = validation$Class,  thresholds=c(seq(0.001, 0.999, 0.002)), metric = "Youden", plot = F)
      threshold_Youden_optimal_instance <- threshold_Youden_optimal_instance$optimal_threshold
    
      threshold_F1_optimal_logit <- findOptimalThreshold(scores = scores_logit_validation, true_classes = validation$Class, thresholds=c(seq(0.001, 0.999, 0.002)), metric = "F1", plot = F)
      threshold_F1_optimal_logit <- threshold_F1_optimal_logit$optimal_threshold
    
      threshold_F1_optimal_class <- findOptimalThreshold(scores = scores_class_logit_validation, true_classes = validation$Class, thresholds=c(seq(0.001, 0.999, 0.002)), metric = "F1", plot = F)
      threshold_F1_optimal_class <- threshold_F1_optimal_class$optimal_threshold
    
      threshold_F1_optimal_instance <- findOptimalThreshold(scores =  scores_instance_logit_validation, true_classes = validation$Class,  thresholds=c(seq(0.001, 0.999, 0.002)), metric = "F1", plot = F)
      threshold_F1_optimal_instance <- threshold_F1_optimal_instance$optimal_threshold
    
      threshold_Savings_optimal_logit <- findOptimalThreshold(scores = scores_logit_validation, true_classes = validation$Class, thresholds=c(seq(0.001, 0.999, 0.002)), metric = "Savings", cost_matrix = instance_matrix_validation,  plot = F)
      threshold_Savings_optimal_logit <- threshold_Savings_optimal_logit$optimal_threshold
    
      threshold_Savings_optimal_class <- findOptimalThreshold(scores = scores_class_logit_validation, true_classes = validation$Class,  thresholds=c(seq(0.001, 0.999, 0.002)), metric = "Savings", cost_matrix = instance_matrix_validation,  plot = F)
      threshold_Savings_optimal_class <- threshold_Savings_optimal_class$optimal_threshold
    
      threshold_Savings_optimal_instance <- findOptimalThreshold(scores =  scores_instance_logit_validation, true_classes = validation$Class,  thresholds=c(seq(0.001, 0.999, 0.002)), metric = "Savings", cost_matrix = instance_matrix_validation,  plot = F)
      threshold_Savings_optimal_instance <- threshold_Savings_optimal_instance$optimal_threshold
    
      thresholds_constant <- rbind.data.frame(thresholds_constant, cbind.data.frame(k_run, j_fold,
                                                                 threshold_0.5,
                                                                 threshold_imbalance_ratio,
                                                                 threshold_class_cost,
                                                                 threshold_Youden_optimal_logit,
                                                                 threshold_Youden_optimal_class,
                                                                 threshold_Youden_optimal_instance,
                                                                 threshold_F1_optimal_logit,
                                                                 threshold_F1_optimal_class,
                                                                 threshold_F1_optimal_instance,
                                                                 threshold_Savings_optimal_logit,
                                                                 threshold_Savings_optimal_class,
                                                                 threshold_Savings_optimal_instance))
    
      threshold_BMR <- rbind.data.frame(threshold_BMR, cbind.data.frame(k_run, j_fold, threshold_instance_cost))
    
      
      ## Performance on test set
      results_logit_threshold_0.5  <- rbind.data.frame(results_logit_threshold_0.5, cbind.data.frame(k_run, j_fold, performanceMeasures(scores_logit_test, threshold_0.5, test$Class, instance_matrix_test)))
      results_class_logit_threshold_0.5 <- rbind.data.frame(results_class_logit_threshold_0.5, cbind.data.frame(k_run, j_fold, performanceMeasures(scores_class_logit_test, threshold_0.5, test$Class, instance_matrix_test)))
      results_instance_logit_threshold_0.5 <- rbind.data.frame(results_instance_logit_threshold_0.5, cbind.data.frame(k_run, j_fold, performanceMeasures(scores_instance_logit_test, threshold_0.5, test$Class, instance_matrix_test)))
    
      results_logit_threshold_imbalance_ratio  <- rbind.data.frame(results_logit_threshold_imbalance_ratio, cbind.data.frame(k_run, j_fold, performanceMeasures(scores_logit_test, threshold_imbalance_ratio, test$Class, instance_matrix_test)))
      results_class_logit_threshold_imbalance_ratio <- rbind.data.frame(results_class_logit_threshold_imbalance_ratio, cbind.data.frame(k_run, j_fold, performanceMeasures(scores_class_logit_test, threshold_imbalance_ratio, test$Class, instance_matrix_test)))
      results_instance_logit_threshold_imbalance_ratio <- rbind.data.frame(results_instance_logit_threshold_imbalance_ratio, cbind.data.frame(k_run, j_fold, performanceMeasures(scores_instance_logit_test, threshold_imbalance_ratio, test$Class, instance_matrix_test)))
    
      results_logit_threshold_class_cost  <- rbind.data.frame(results_logit_threshold_class_cost, cbind.data.frame(k_run, j_fold, performanceMeasures(scores_logit_test, threshold_class_cost, test$Class, instance_matrix_test)))
      results_class_logit_threshold_class_cost <- rbind.data.frame(results_class_logit_threshold_class_cost, cbind.data.frame(k_run, j_fold, performanceMeasures(scores_class_logit_test, threshold_class_cost, test$Class, instance_matrix_test)))
      results_instance_logit_threshold_class_cost <- rbind.data.frame(results_instance_logit_threshold_class_cost, cbind.data.frame(k_run, j_fold, performanceMeasures(scores_instance_logit_test, threshold_class_cost, test$Class, instance_matrix_test)))
    
      results_logit_threshold_instance_cost  <- rbind.data.frame(results_logit_threshold_instance_cost, cbind.data.frame(k_run, j_fold, performanceMeasures(scores_logit_test, threshold_instance_cost, test$Class, instance_matrix_test)))
      results_class_logit_threshold_instance_cost <- rbind.data.frame(results_class_logit_threshold_instance_cost, cbind.data.frame(k_run, j_fold, performanceMeasures(scores_class_logit_test, threshold_instance_cost, test$Class, instance_matrix_test)))
      results_instance_logit_threshold_instance_cost <- rbind.data.frame(results_instance_logit_threshold_instance_cost, cbind.data.frame(k_run, j_fold, performanceMeasures(scores_instance_logit_test, threshold_instance_cost, test$Class, instance_matrix_test)))
    
      results_logit_threshold_Youden_optimal  <- rbind.data.frame(results_logit_threshold_Youden_optimal, cbind.data.frame(k_run, j_fold, performanceMeasures(scores_logit_test, threshold_Youden_optimal_logit, test$Class, instance_matrix_test)))
      results_class_logit_threshold_Youden_optimal <- rbind.data.frame(results_class_logit_threshold_Youden_optimal, cbind.data.frame(k_run, j_fold, performanceMeasures(scores_class_logit_test, threshold_Youden_optimal_class, test$Class, instance_matrix_test)))
      results_instance_logit_threshold_Youden_optimal <- rbind.data.frame(results_instance_logit_threshold_Youden_optimal, cbind.data.frame(k_run, j_fold, performanceMeasures(scores_instance_logit_test, threshold_Youden_optimal_instance, test$Class, instance_matrix_test)))
    
      results_logit_threshold_F1_optimal  <- rbind.data.frame(results_logit_threshold_F1_optimal, cbind.data.frame(k_run, j_fold, performanceMeasures(scores_logit_test, threshold_F1_optimal_logit, test$Class, instance_matrix_test)))
      results_class_logit_threshold_F1_optimal <- rbind.data.frame(results_class_logit_threshold_F1_optimal, cbind.data.frame(k_run, j_fold, performanceMeasures(scores_class_logit_test, threshold_F1_optimal_class, test$Class, instance_matrix_test)))
      results_instance_logit_threshold_F1_optimal <- rbind.data.frame(results_instance_logit_threshold_F1_optimal, cbind.data.frame(k_run, j_fold, performanceMeasures(scores_instance_logit_test, threshold_F1_optimal_instance, test$Class, instance_matrix_test)))
    
      results_logit_threshold_Savings_optimal  <- rbind.data.frame(results_logit_threshold_Savings_optimal, cbind.data.frame(k_run, j_fold, performanceMeasures(scores_logit_test, threshold_Savings_optimal_logit, test$Class, instance_matrix_test)))
      results_class_logit_threshold_Savings_optimal <- rbind.data.frame(results_class_logit_threshold_Savings_optimal, cbind.data.frame(k_run, j_fold, performanceMeasures(scores_class_logit_test, threshold_Savings_optimal_class, test$Class, instance_matrix_test)))
      results_instance_logit_threshold_Savings_optimal <- rbind.data.frame(results_instance_logit_threshold_Savings_optimal, cbind.data.frame(k_run, j_fold, performanceMeasures(scores_instance_logit_test, threshold_Savings_optimal_instance, test$Class, instance_matrix_test)))
    }
  
    
    if( nruns == 25){
      ## Regression summaries
      logit_AIC_R2 <- rbind.data.frame(logit_AIC_R2, cbind.data.frame(k_run, j_fold, stats::AIC(logit), DescTools::PseudoR2(logit)))
      
      # name_of_file_5 <- paste("~/Documents/KU Leuven/Master's Thesis/Data/Results/",resample_setting , "_run_", k_run, "_fold_", j_fold, "_logit_summary", ".txt", sep="", collapse=NULL)
      # sink(file = name_of_file_5)
      # print(summary(logit))
      # sink()
      # name_of_file_5 <- paste("~/Documents/KU Leuven/Master's Thesis/Data/Results/",resample_setting , "_run_", k_run, "_fold_", j_fold, "_class_summary", ".txt", sep="", collapse=NULL)
      # sink(file = name_of_file_5)
      # print(summary(class_logitL1))
      # sink()
      # name_of_file_5 <- paste("~/Documents/KU Leuven/Master's Thesis/Data/Results/",resample_setting , "_run_", k_run, "_fold_", j_fold, "_instance_summary", ".txt", sep="", collapse=NULL)
      # sink(file = name_of_file_5)
      # print(summary(instance_logitL1))
      # sink()
        
      
      ## Prediction on test set
      prediction_logit <- rbind.data.frame(prediction_logit, cbind.data.frame(k_run, j_fold, scores_logit_test))
      prediciton_class <- rbind.data.frame(prediciton_class, cbind.data.frame(k_run, j_fold, scores_class_logit_test))
      prediciton_instance <- rbind.data.frame(prediciton_instance, cbind.data.frame(k_run, j_fold, scores_instance_logit_test))
      
      
      ### POST-PROCESSING----------------------------------------------------------
      
      
      ## Performance on test set (THRESHOLD = 0.5)
      results_logit_threshold_0.5  <- rbind.data.frame(results_logit_threshold_0.5, cbind.data.frame(k_run, j_fold, performanceMeasures(scores_logit_test, 0.5, test$Class, instance_matrix_test)))
      results_class_logit_threshold_0.5 <- rbind.data.frame(results_class_logit_threshold_0.5, cbind.data.frame(k_run, j_fold, performanceMeasures(scores_class_logit_test, 0.5, test$Class, instance_matrix_test)))
      results_instance_logit_threshold_0.5 <- rbind.data.frame(results_instance_logit_threshold_0.5, cbind.data.frame(k_run, j_fold, performanceMeasures(scores_instance_logit_test, 0.5, test$Class, instance_matrix_test)))
      
      
      ## Performance on train set (THRESHOLD = 0.5)
      results_logit_threshold_0.5_train  <- rbind.data.frame(results_logit_threshold_0.5_train, cbind.data.frame(k_run, j_fold, performanceMeasures(fitted_values_logit_run, 0.5, train$Class, insensitive_matrix_train)))
      results_class_logit_threshold_0.5_train <- rbind.data.frame(results_class_logit_threshold_0.5_train, cbind.data.frame(k_run, j_fold, performanceMeasures(class_logitL1$fitted_values, 0.5, train$Class, class_matrix_train)))
      results_instance_logit_threshold_0.5_train <- rbind.data.frame(results_instance_logit_threshold_0.5_train, cbind.data.frame(k_run, j_fold, performanceMeasures(instance_logitL1$fitted_values, 0.5, train$Class, instance_matrix_train)))
    }
  }
}


if(nruns == 5){
  ### Methods names 
  methods <- c("logit_threshold_0.5", "class_logit_threshold_0.5", "instance_logit_threshold_0.5", 
               "logit_threshold_imbalance_ratio", "class_logit_threshold_imbalance_ratio", "instance_logit_threshold_imbalance_ratio", 
               "logit_threshold_class_cost", "class_logit_threshold_class_cost", "instance_logit_threshold_class_cost",
               "logit_threshold_instance_cost", "class_logit_threshold_instance_cost", "instance_logit_threshold_instance_cost",
               "logit_threshold_Youden_optimal","class_logit_threshold_Youden_optimal", "instance_logit_threshold_Youden_optimal",
               "logit_threshold_F1_optimal","class_logit_threshold_F1_optimal", "instance_logit_threshold_F1_optimal",
               "logit_threshold_Savings_optimal","class_logit_threshold_Savings_optimal", "instance_logit_threshold_Savings_optimal")
  
  methods <- paste(resample_setting, methods, sep="_")
  methods_vec <- rep(methods, nruns * nfolds)[order(match(rep(methods, nruns * nfolds), methods))]
  
  
  ### Results
  results <- rbind.data.frame(results_logit_threshold_0.5, results_class_logit_threshold_0.5, results_instance_logit_threshold_0.5,
                              results_logit_threshold_imbalance_ratio, results_class_logit_threshold_imbalance_ratio, results_instance_logit_threshold_imbalance_ratio,
                              results_logit_threshold_class_cost,results_class_logit_threshold_class_cost, results_instance_logit_threshold_class_cost,
                              results_logit_threshold_instance_cost, results_class_logit_threshold_instance_cost, results_instance_logit_threshold_instance_cost, 
                              results_logit_threshold_Youden_optimal, results_class_logit_threshold_Youden_optimal, results_instance_logit_threshold_Youden_optimal,
                              results_logit_threshold_F1_optimal, results_class_logit_threshold_F1_optimal, results_instance_logit_threshold_F1_optimal,
                              results_logit_threshold_Savings_optimal, results_class_logit_threshold_Savings_optimal, results_instance_logit_threshold_Savings_optimal)
  
  results <- cbind.data.frame(results, method = methods_vec)
  
  results$PreProcessing <- resample_setting
  nrep <- nruns * nfolds
  results$Processing <- c(rep(c(rep("logit",nrep), rep("class",nrep), rep("instance",nrep)),7))
  results$PostProcessing <- c(rep("threshold_0.5",3*nrep), 
                              rep("threshold_imbalance_ratio",3*nrep), 
                              rep("threshold_class_cost", 3*nrep), 
                              rep("threshold_instance_cost", 3*nrep), 
                              rep("threshold_Youden_optimal_logit",nrep), 
                              rep("threshold_Youden_optimal_class",nrep), 
                              rep("threshold_Youden_optimal_instance",nrep), 
                              rep("threshold_F1_optimal_logit",nrep),
                              rep("threshold_F1_optimal_class",nrep),
                              rep("threshold_F1_optimal_instance",nrep), 
                              rep("threshold_Savings_optimal_logit", nrep), 
                              rep("threshold_Savings_optimal_class", nrep),
                              rep("threshold_Savings_optimal_instance", nrep))
  
  if (resample_setting %in% c("ROS", "RUS", "RAW")){
    results$CostSensitive <- c(rep("No", nrep), rep("Yes", 2*nrep), rep("No", nrep), rep("Yes", 8*nrep), rep("No", nrep), rep("Yes",2*nrep), rep("No",nrep), rep("Yes",5*nrep))
  } else {
    results$CostSensitive <- c(rep("Yes", 7*3*nrep))
  }
  
  results <- results[, c(34:38, 1:33)]
  
  
  ### Coefficients
  coefficients_logit$Processing <- "logit"
  coefficients_class$Processing <- "class"
  coefficients_instance$Processing <- "instance"
  
  coefficients_methods <- rbind.data.frame(coefficients_logit, coefficients_class, coefficients_instance)
  coefficients_methods$PreProcessing <- resample_setting
  coefficients_methods <- coefficients_methods[, c(34:33, 1:32)] # rearrange columns
  
  
  ### Fitted values
  fitted_values_logit$Processing <- "logit"
  fitted_values_class$Processing <- "class"
  fitted_values_instance$Processing <- "instance"
  
  colnames(fitted_values_logit)[3] <- "fitted_value"
  colnames(fitted_values_class)[3] <- "fitted_value"
  colnames(fitted_values_instance)[3] <- "fitted_value"
  colnames(fitted_values_logit)[4] <- "Class"
  colnames(fitted_values_class)[4] <- "Class"
  colnames(fitted_values_instance)[4] <- "Class"
  
  fitted_values_methods <- rbind.data.frame(fitted_values_logit, fitted_values_class, fitted_values_instance)
  fitted_values_methods$PreProcessing <- resample_setting
  fitted_values_methods <- fitted_values_methods[, c(1:2, 6:5, 4:3)]
  
  
  ### Testset and prediction
  testset_with_prediction_logit$Processing <- "logit"
  testset_with_prediction_class$Processing <- "class"
  testset_with_prediction_instance$Processing <-"instance"
  
  colnames(testset_with_prediction_logit)[33]<-"scores" 
  colnames(testset_with_prediction_class)[33]<-"scores"
  colnames(testset_with_prediction_instance)[33]<-"scores"
  
  testset_with_prediction_methods <- rbind.data.frame(testset_with_prediction_logit, testset_with_prediction_class, testset_with_prediction_instance)
  testset_with_prediction_methods$PreProcessing <- resample_setting
  testset_with_prediction_methods <- testset_with_prediction_methods[, c(35:34, 1:33)]
  
  
  ### Thresholds
  thresholds_constant
  average_thresholds_constant <- colMeans(thresholds_constant)
  
  ### Average results
  ## Average results with all performances
  average_results <- aggregate(results, by = list(methods_vec), FUN = mean, na.rm = TRUE)
  average_results <- average_results[order(match(average_results$Group.1, methods)), ] 
  ## Average results with main performances and Processing, PostProcessing, CostSensitive columns
  results$method
  average_rank <- summarise(group_by(results, method), 
                                  Savings = mean(Savings, na.rm = TRUE),
                                  F1 = mean(F1, na.rm = TRUE),
                                  AUC = mean (AUC, na.rm=TRUE),
                                  PreProcessing=unique(PreProcessing),
                                  Processing = unique(Processing),
                                  PostProcessing = unique(PostProcessing),
                                  CostSensitive = unique(CostSensitive))
  average_rank <- ungroup(average_rank)
  average_rank <- average_rank[order(match(average_rank$method, unique(results$method))), ]
  average_rank$SavingsRank <- dense_rank(-(average_rank$Savings))
  average_rank$F1Rank <- dense_rank(-(average_rank$F1))
  average_rank$AUCRank <- dense_rank(-(average_rank$AUC))
  average_rank <- select(average_rank, "method", "PreProcessing", "Processing", "PostProcessing", "CostSensitive", "Savings", "SavingsRank", "F1", "F1Rank", "AUC", "AUCRank")
  average_rank <- arrange(average_rank, desc(Savings))

  
  ### Median results
  median_results <- dplyr::filter( group_by(results, method), Savings==Savings[which.min(abs((median(Savings)-0.0001)-Savings))])
  median_results <- median_results[order(match(median_results$method, unique(results$method))), ]
  
  ### Number of instances
  number_instance <- rbind.data.frame(number_instance_prior, number_instance_post)
}


if(nruns == 25){
  ### Results
  results_0.5_train <- rbind.data.frame(results_logit_threshold_0.5_train, results_class_logit_threshold_0.5_train, results_instance_logit_threshold_0.5_train)
  results_0.5_train$PreProcessing <- resample_setting
  nrep <- nruns * nfolds
  results_0.5_train$Processing <- c(rep("logit", nrep), rep("class", nrep), rep("instance", nrep))
  results_0.5_train <- results_0.5_train[,c(35, 34, 1:33)]
  
  results_0.5_test <- rbind.data.frame(results_logit_threshold_0.5, results_class_logit_threshold_0.5, results_instance_logit_threshold_0.5)
  results_0.5_test$PreProcessing <- resample_setting
  results_0.5_test$Processing <- c(rep("logit", nrep), rep("class", nrep), rep("instance", nrep))
  results_0.5_test <- results_0.5_test[,c(35, 34, 1:33)]

  
  ### Prediction test 
  prediction_logit$Processing <- "logit"
  prediciton_class$Processing <- "class"
  prediciton_instance$Processing <- "instance"
  colnames(prediction_logit)[3] <- "prediction"
  colnames(prediciton_class)[3] <- "prediction"
  colnames(prediciton_instance)[3] <- "prediction"
  
  prediction_methods <- rbind.data.frame(prediction_logit, prediciton_class, prediciton_instance)
  prediction_methods <- prediction_methods[,3:4]
  
  
  ### Coefficient analysis (Pre-settings: nruns = 25, resampling_setting = "RAW")
  if (resample_setting == "RAW"){
    ### Logit
    coef_logit <- matrix(unlist(coefficients_logit[ , 3:32]), ncol = 30, byrow = FALSE)
    colnames(coef_logit) <- colnames(coefficients_logit[ , 3:32])
    summary(coef_logit)
    
    coef_logit <-data.frame(coef_logit)
    mean_coef_logit <- colMeans(coef_logit)
    std_dev_coef_logit <- colSds(coef_logit)
    
    stats_logit <- rbind(mean_coef_logit, std_dev_coef_logit)
    colnames(stats_logit) <- colnames(coef_logit)
    stats_logit <- rbind(coef_logit, stats_logit)
    
    ### CSLogit class
    coef_class <- matrix(unlist(coefficients_class[ , 3:32]), ncol = 30, byrow = FALSE)
    colnames(coef_class) <- colnames(coefficients_class[ , 3:32])
    summary(coef_class)
    
    coef_class <-data.frame(coef_class)
    mean_coef_class <- colMeans(coef_class)
    std_dev_coef_class <- colSds(coef_class)
    
    stats_class <- rbind(mean_coef_class, std_dev_coef_class)
    colnames(stats_class) <- colnames(coef_class)
    stats_class <- rbind(coef_class, stats_class)
    
    ### CSLogit instance
    coef_instance <- matrix(unlist(coefficients_instance[ , 3:32]), ncol = 30, byrow = FALSE)
    colnames(coef_instance) <- colnames(coefficients_instance[ , 3:32])
    summary(coef_instance)
    
    coef_instance <-data.frame(coef_instance)
    mean_coef_instance <- colMeans(coef_instance)
    std_dev_coef_instance <- colSds(coef_instance)
    
    stats_instance <- rbind(mean_coef_instance, std_dev_coef_instance)
    colnames(stats_instance) <- colnames(coef_instance)
    stats_instance <- rbind(coef_instance, stats_instance)
    
    
    ### Plot the relationship between LogAmount and P(Fraudulent)
    ## Input data
    sample <- train
    nrep <- nruns * nfolds
    coefficient_means <- rbind(stats_logit[nrep+1, ], stats_class[nrep+1, ], stats_instance[nrep+1, ]) # n=1 logit coefficent mean; n=2 class coefficient mean; n=3 instance coefficient mean
    
    
    ## Plot data
    betas_amount_range <- seq(from=min(sample$LogAmount), to=max(sample$LogAmount), by=.001)
    betas_mean_sample <- colMeans(sample[,1:28])
    betas_new_sample <- cbind.data.frame(1, t(betas_mean_sample[1:28]))
    betas_new_sample <- cbind.data.frame(betas_new_sample, betas_amount_range)
    betas_new_sample <-  matrix(unlist(betas_new_sample), ncol = 30, byrow = FALSE)
    colnames(betas_new_sample) <- colnames(paste(coefficient_means[ , 1:29], "LogAmount_range"))
    betas <- matrix(unlist(coefficient_means[ , ]), ncol = 30, byrow = FALSE)
    betas_plot_matrix <- matrix(nrow = nrow(betas_new_sample), ncol = 1)
    
    for (i in 1:nrow(betas)) {
      betas_amount_logit_z <- c()
      betas_amount_logit <- c()
      betas_amount_probs <- c()
      for (z in 1:nrow(betas_new_sample)) {
        betas_amount_logit_z <- betas[i, ] %*% betas_new_sample[z, ]
        betas_amount_logit <- rbind.data.frame(betas_amount_logit, betas_amount_logit_z)
      }
      betas_amount_logit <- matrix(unlist(betas_amount_logit), ncol = 1, byrow = FALSE)
      betas_amount_probs <- exp(betas_amount_logit)/(1 + exp(betas_amount_logit))
      betas_plot_matrix <- cbind(betas_plot_matrix,  cbind(i, betas_amount_range, betas_amount_probs))
    }
    
    betas_plot_matrix <- betas_plot_matrix[ , 2:ncol(betas_plot_matrix)]
    
    
    ## Plots
    par(mfrow = c(1, 3))
    v <- 2
    w <- ncol(betas_plot_matrix)
    q <- 1
    repeat{
      plot(betas_plot_matrix [,v], betas_plot_matrix[,v+1], xlim=c(min(betas_plot_matrix [,v])-1, max(betas_plot_matrix [,v])+1), ylim=c(0, max(betas_plot_matrix [,v+1])*1.25), type="l", lwd=3, lty=1, col=q, xlab="LogAmount", ylab="P(Fraudulent)")
      v <- v + 3
      w <- w + 1
      q <- q + 1
      if(v > w)
        break
    }
    
    legend("topright", c("CI","CDCS","IDCS"), fill = c((q-3),(q-2),(q-1))) # to add only if 3 plots concerns CI, CDCS and IDCS logistic regressions
    mtext("Relationship between LogAmount and probability of being fraudulent", outer=TRUE,  cex=1.5, line=-2)
    par(mfrow = c(1, 1))
  }  
}


### Print the results 
## nruns = 5
average_rank
# results
# average_results
# median_results
# coefficients_methods
# testset_with_prediction_methods
# thresholds_constant
# threshold_BMR
# fitted_values_methods
# number_instance
# amount_test_cv


## nruns = 25
# fitted_values_methods
# results_logit_train
# results_logit_test
# prediction_methods
# logit_AIC_R2



# Uncomment lines 759-1535 to run the plot analysis

# ###Plots Analysis---------------------------------------------------------------

# ### Packages
# library(rstatix)
# library(readxl)
# library(tidyverse)
# library(xlsx)
# library(Amelia)
# library(mlbench)
# library(Hmisc)
# library(geometry)
# library(ROCR)
# library(ggrepel)
# library(ggforce)
# library(robustHD)
# library(caTools)
# library(cslogit)
# library(ggplot2)
# library(xgboost)
# library(gridExtra)
# library(lubridate)
# library(splitstackshape)
# library(dplyr)
# library(PerformanceMetrics)
# library(tidyverse)
# library(ROCR)
# library(data.table)
# 
# 
# ### Best CS and CI methods on savings-------------------------------------------
# ## best CI method
# average_rank_cost_insensitive<- dplyr::filter(average_rank, CostSensitive=="No")
# max_average_Savings <- max(average_rank_cost_insensitive$Savings)
# best_method_cost_insensitive <- dplyr::filter(average_rank_cost_insensitive, Savings==max_average_Savings)
# best_method_cost_insensitive <- best_method_cost_insensitive$method
# 
# 
# ## best CS method
# average_rank_cost_sensitive <- dplyr::filter(average_rank, CostSensitive=="Yes")
# max_average_Savings <- max(average_rank_cost_sensitive$Savings)
# best_method_cost_sensitive <- dplyr::filter(average_rank_cost_sensitive, Savings==max_average_Savings)
# best_method_cost_sensitive <- best_method_cost_sensitive$method
# 
# 
# ## results best methods
# results_best <- dplyr::filter(results, method==best_method_cost_insensitive| method==best_method_cost_sensitive)
# results_best$method <- c(rep("CDCS Instance cost", 10), rep("CI F1 optimal", 10))
# 
# 
# ## average results best methods
# average_rank_best <- dplyr::filter(average_rank, method==best_method_cost_insensitive| method==best_method_cost_sensitive)
# 
# 
# ## Boxplot function
# createBoxplots <- function (df, ylabel, ylimit, average_measures) {
#   fontfaces <- rep("plain", length(unique(df$method)))
#   fontfaces[which.max(average_measures)] <- "bold"
#   if (any(average_measures > 100) | grepl("time", ylabel)) {
#     label_average <- round(average_measures, 2)
#   } else {
#     label_average <- paste0(round(average_measures, 2), "%")
#   }
#   boxplots <- ggplot(data = df, mapping = aes(x = method, y = measure, fill = method)) +
#     stat_boxplot(geom = "errorbar", width = 0.4) + geom_boxplot() +
#     ylab(ylabel) + xlab("") + ylim(ylimit) + scale_x_discrete(limits = unique(df$method)) + 
#     stat_summary(fun = mean, geom = "point", shape = 18, size = 5, col = "black") +
#     geom_text(data = data.frame(method = unique(df$method), average_measures = average_measures),
#               mapping = aes(x = method, y = min(ylimit, na.rm = TRUE),
#                             label = label_average), size = 5,
#               fontface = fontfaces) +
#     theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
#           panel.background = element_blank(), axis.line = element_line(colour = "black"))+
#     theme(text = element_text(size = 15), legend.position = "none", axis.text.x = element_text(angle = 0))
#   return(boxplots)
# }
# 
# 
# ## Boxplot on savings
# box_savings_best <- createBoxplots(ylabel = "Savings (%)",
#                                    df = data.frame(measure = 100 * results_best$Savings,
#                                                    method  = results_best$method),
#                                    ylimit = c(100 * min(results_best$Savings, na.rm = TRUE) - 10,
#                                               min(100 * max(results_best$Savings, na.rm = TRUE) + 10, 100)),
#                                    average_measures = 100 * average_rank_best$Savings)
# 
# 
# 
# 
# 
# ### Results: CI, CDCS, IDCS, threshold 0.5--------------------------------------
# ## Results with 0.5
# results_0.5 <- dplyr::filter(results, Processing %in% c("logit", "class", "instance"), PostProcessing=="threshold_0.5")
# results_0.5$method <- c(rep("CI", 10), rep("CDCS", 10), rep("IDCS", 10))
# 
# 
# ## Average results with 0.5
# average_0.5 <- dplyr::filter(average_rank,  Processing %in% c("logit", "class", "instance"), PostProcessing=="threshold_0.5")
# average_0.5 <- arrange(average_0.5, -row_number())
# 
# 
# ## Boxplots
# average_0.5_boxplot_savings <- createBoxplots(ylabel = "Savings (%)",
#                                               df = data.frame(measure = 100 * results_0.5$Savings,
#                                                               method  = results_0.5$method),
#                                               ylimit = c(0, 100),
#                                               average_measures = 100 * average_0.5$Savings)
# 
# average_0.5_boxplot_AUC <- createBoxplots(ylabel = "AUC (%)",
#                                           df = data.frame(measure = 100 * results_0.5$AUC,
#                                                           method  = results_0.5$method),
#                                           ylimit = c(80, 100),
#                                           average_measures = 100 * average_0.5$AUC)
# 
# average_0.5_boxplot_F1 <- createBoxplots(ylabel = "F1 (%)",
#                                          df = data.frame(measure = 100 * results_0.5$F1,
#                                                          method  = results_0.5$method),
#                                          ylimit = c(0, 100),
#                                          average_measures = 100 * average_0.5$F1)
# 
# 
# 
# 
# 
# ### Results all combinations----------------------------------------------------
# ## Average results
# average_savings_results_for_boxplots <- results %>%
#   group_by(method)%>%
#   summarise(Savings = mean(Savings, na.rm = TRUE),
#             F1 = mean(F1, na.rm = TRUE),
#             AUC = mean(AUC, na.rm = TRUE))
# average_savings_results_for_boxplots <- arrange(average_savings_results_for_boxplots,desc(Savings))
# average_savings <- average_savings_results_for_boxplots$Savings
# average_F1 <-average_savings_results_for_boxplots$F1
# average_AUC <- average_savings_results_for_boxplots$AUC
# average_savings_results_for_boxplots <- lapply(average_savings_results_for_boxplots$method, rep, 10)
# average_savings_results_for_boxplots <- unlist(average_savings_results_for_boxplots)
# average_savings_results_for_boxplots <- as.vector(average_savings_results_for_boxplots)
# 
# results_for_boxplots <- results%>%
#   arrange(factor(method, levels=unique(average_savings_results_for_boxplots)))
# 
# 
# ## New Boxplot function
# library(ggplot2)
# library(gridExtra)
# 
# createBoxplots <- function (df, ylabel, ylimit, average_measures) {
#   fontfaces <- rep("plain", length(unique(df$method)))
#   fontfaces[which.max(average_measures)] <- "bold"
#   if (any(average_measures > 100) | grepl("time", ylabel)) {
#     label_average <- round(average_measures, 2)
#   } else {
#     label_average <- paste0(round(average_measures, 2), "%")
#   }
#   boxplots <- ggplot(data = df, mapping = aes(x = method, y = measure, fill = method)) +
#     stat_boxplot(geom = "errorbar", width = 0.4) + geom_boxplot() +
#     ylab(ylabel) + xlab("") + ylim(ylimit) + scale_x_discrete(limits = unique(df$method), label=method_boxplots)+
#     stat_summary(fun = mean, geom = "point", shape = 18, size = 5, col = "black") +
#     geom_text(data = data.frame(method = unique(df$method), average_measures = average_measures),
#               mapping = aes(x = method, y = min(ylimit, na.rm = TRUE),
#                             label = label_average), size = 6.5,
#               fontface = fontfaces) +
#     theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
#           panel.background = element_blank(), axis.line = element_line(colour = "black"))+
#     theme(text = element_text(size = 18), legend.position = "none", axis.text.x = element_text(angle = 0))
#   return(boxplots)
# }
# 
# method_boxplots <- c(paste("CDCS\n Instance cost"), paste("IDCS\n Instance cost"), paste("CDCS\n Savings optimal"), paste("CDCS\n F1 optimal"), 
#                      paste("IDCS\n F1 optimal"), paste("IDCS\n Savings optimal"), paste("IDCS\n Youden optimal"), paste("IDCS\n 0.5"), paste("IDCS\n Class cost"), 
#                      paste("CDCS\n 0.5"), paste("CDCS\n Youden optimal"), paste("CI\n Instance cost"), paste("CDCS\n Class cost"), paste("CI\n Savings optimal"), 
#                      paste("CI\n Class cost"), paste("CI\n F1 optimal"), paste("IDCS\n Imbalance ratio"), paste("CI\n 0.5"), paste("CDCS\n Imbalance ratio"), 
#                      paste("CI\n Youden optimal"), paste("CI\n Imbalance ratio"))
# 
# all_results_boxplot_Savings <- createBoxplots(ylabel = "Savings (%)",
#                                               df = data.frame(measure = 100 * results_for_boxplots$Savings,
#                                                               method  = results_for_boxplots$method),
#                                               ylimit = c(-200, 100),
#                                               average_measures = 100 * average_savings)
# 
# all_results_boxplot_F1 <- createBoxplots(ylabel = "F1 (%)",
#                                          df = data.frame(measure = 100 * results_for_boxplots$F1,
#                                                          method  = results_for_boxplots$method),
#                                          ylimit = c(0, 100),
#                                          average_measures = 100 * average_F1)
# 
# all_results_boxplot_AUC <- createBoxplots(ylabel = "AUC (%)",
#                                           df = data.frame(measure = 100 * results_for_boxplots$AUC,
#                                                           method  = results_for_boxplots$method),
#                                           ylimit = c(80, 100),
#                                           average_measures = 100 * average_AUC)
# 
# 
# 
# 
# 
# #### ROC Curve, Precision-recall Curve, Scores Visualisation, Confusion Matrixes ----------------------------
# ## ROC CI
# results_logit<- dplyr::filter(results, Processing=="logit")
# median_method_AUC <- dplyr::filter(results_logit, AUC==AUC[which.min(abs((median(results_logit$AUC)-0.00001)-AUC))])
# median_Processing <- unique(median_method_AUC$Processing)
# median_k_run <- unique(median_method_AUC$k_run)
# median_j_fold <- unique(median_method_AUC$j_fold)
# 
# AUC_logit <- round(unique(median_method_AUC$AUC),3)
# 
# testset_with_prediction_logit <- dplyr::filter(testset_with_prediction_methods, Processing==median_Processing, k_run==median_k_run, j_fold==median_j_fold)
# testset_with_prediction_logit
# 
# tpr_fpr_precision_recall_logit <- select(median_method_AUC, TPR, FPR, Precision, Recall)
# tpr_fpr_precision_recall_logit$Savings <- median_method_AUC$Savings
# thresholds_names <- c("0.5","Imbalance ratio","Class cost", "Instance cost", "Youden optimal", "F1 optimal", "Savings optimal")
# tpr_fpr_precision_recall_logit$thresholds <- thresholds_names
# 
# tpr_fpr_logit <-select(tpr_fpr_precision_recall_logit,TPR, FPR, Savings, thresholds)
# precision_recall_logit <- select(tpr_fpr_precision_recall_logit, Precision, Recall, Savings, thresholds)
# 
# pr <- prediction(testset_with_prediction_logit$scores, testset_with_prediction_logit$Class) #Ok because we haven't specified the threshold, the ROC curve is drawn for all thresholds
# perf <- ROCR::performance(pr, measure = "tpr", x.measure = "fpr") #Ok the performance function is used to draw the ROC curve
# roc_dt <- data.frame( fpr = perf@x.values[[1]], tpr = perf@y.values[[1]] )
# 
# logit_roc <- ggplot( roc_dt, aes( fpr, tpr ), label=Name) + 
#   geom_line( color = rgb( 0, 0, 1, alpha = 0.8 ) ) +
#   labs( title = paste("ROC: CI", "( AUC =",AUC_logit,")"), x = "False Postive Rate", y = "True Positive Rate" )+
#   geom_point( data=tpr_fpr_logit, aes(x=FPR, y= TPR, color=Savings), size = 2, alpha = 0.8 ) +
#   geom_label_repel(data=tpr_fpr_logit, mapping=aes(x=FPR, y= TPR, label = thresholds),
#                    box.padding   = 0.35, 
#                    point.padding = 0.5,
#                    nudge_x = 0.32,
#                    nudge_y = -0.05,
#                    force = 1.2,
#                    segment.color = 'grey50') +
#   scale_color_gradient(low="red", high="green")+
#   theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
#         panel.background = element_blank(), axis.line = element_line(colour = "black"))
# 
# 
# ## Precision-recall CI
# perf <- ROCR::performance(pr, "prec","rec") 
# precision_recall_dt <- data.frame( rec = perf@x.values[[1]], prec = perf@y.values[[1]] )
# logit_precision_recall <- ggplot( precision_recall_dt, aes( rec, prec ), label=Name) + 
#   geom_line( color = rgb( 0, 0, 1, alpha = 0.8 ) ) +
#   geom_point( data=precision_recall_logit, aes(x=Recall, y= Precision, color=Savings), size = 2, alpha = 0.8 ) +
#   labs( title = "Precision-Recall: logit", x = "Recall", y = "Precision" )+
#   geom_label_repel(data=precision_recall_logit, mapping=aes(x=Recall, y= Precision, label = thresholds),
#                    box.padding   = 0.35, 
#                    point.padding = 0.5,
#                    nudge_x = 0.32,
#                    nudge_y = -0.05,
#                    force = 1.2,
#                    segment.color = 'grey50' ) +
#   scale_color_gradient(low="red", high="green")+
#   theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
#         panel.background = element_blank(), axis.line = element_line(colour = "black"))
# 
# 
# ## Confusion matrixes CI
# TP_FP_TN_FN <- select(median_method_AUC, TP, FP, TN, FN)
# TP_FP_TN_FN$thresholds <- thresholds_names
# 
# threshold_BMR_logit <- dplyr::filter(threshold_BMR, k_run==median_k_run, j_fold==median_j_fold )
# 
# cutoff <- 0.5
# 
# predict <- testset_with_prediction_logit$scores
# actual  <- relevel(as.factor(testset_with_prediction_logit$Class), "1")
# 
# result <- cbind.data.frame( actual = actual, predict = predict )
# 
# result$type <- ifelse( predict >= cutoff & actual == 1, "TP",
#                        ifelse( predict >= cutoff & actual == 0, "FP", 
#                                ifelse( predict <  cutoff & actual == 1, "FN", "TN" ) ) )
# result$type <- as.factor(result$type)
# 
# logit_confusion_matrix_constant <- ggplot( result, aes( actual, predict) ) + 
#   geom_violin( fill = "white", color = NA ) +
#   geom_jitter( shape = 1, size= 2.5) + 
#   scale_y_continuous( limits = c( 0, 1 ) ) + 
#   #scale_color_discrete( breaks = c( "TP", "FN", "FP", "TN" ) ) + # ordering of the legend 
#   guides( col = guide_legend( nrow = 2 ) ) +#+ # adjust the legend to have two rows  
#   ggtitle( sprintf( "CI") ) +
#   labs(y = "P(Fraudulent)")+
#   theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
#         panel.background = element_blank(), axis.line = element_line(colour = "black"))
# 
# logit_confusion_matrix_0.5 <- ggplot( result, aes( actual, predict, color = type ) ) + 
#   geom_violin( fill = "white", color = NA ) +
#   geom_jitter( shape = 1, size= 2.5) + 
#   geom_hline( yintercept = cutoff, color = "blue", alpha = 0.6 ) + 
#   scale_y_continuous( limits = c( 0, 1 ) ) + 
#   scale_color_discrete( breaks = c( "TP", "FN", "FP", "TN" )) + # ordering of the legend 
#   guides( col = guide_legend( nrow = 2 ) ) +#+ # adjust the legend to have two rows  
#   ggtitle( sprintf( "Confusion Matrix: CI with 0.5 threshold") ) +
#   labs(y = "P(Fraudulent)")+
#   theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
#         panel.background = element_blank(), axis.line = element_line(colour = "black"))
# 
# cutoff <- threshold_BMR_logit$threshold_instance_cost
# 
# predict <- testset_with_prediction_logit$scores
# actual  <- relevel(as.factor(testset_with_prediction_logit$Class), "1")
# 
# result <- cbind.data.frame( actual = actual, predict = predict )
# 
# result$type <- ifelse( predict >= cutoff & actual == 1, "TP",
#                        ifelse( predict >= cutoff & actual == 0, "FP", 
#                                ifelse( predict <  cutoff & actual == 1, "FN", "TN" ) ) )
# result$type <- as.factor(result$type)
# 
# logit_confusion_matrix_instance <- ggplot( result, aes( actual, predict, color = type ) ) + 
#   geom_violin( fill = "white", color = NA ) +
#   geom_jitter( shape = 1, size= 2.5) + 
#   scale_y_continuous( limits = c( 0, 1 ) ) + 
#   scale_color_discrete( breaks = c( "TP", "FN", "FP", "TN" )) + # ordering of the legend 
#   guides( col = guide_legend( nrow = 2 ) ) +#+ # adjust the legend to have two rows  
#   ggtitle( sprintf( "Confusion Matrix: CI with Instance cost threshold") ) +
#   labs(y = "P(Fraudulent)")+
#   theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
#         panel.background = element_blank(), axis.line = element_line(colour = "black"))
# 
# amount_test_cv_logit <- dplyr::filter(amount_test_cv, k_run==median_k_run, j_fold==median_j_fold)
# amount_test_cv_logit <- select(amount_test_cv_logit, amount_test)
# 
# result$amount_smaller_10 <- ifelse(amount_test_cv_logit<10, "Yes", "No")
# result$amount_smaller_10 <- as.factor(result$amount_smaller_10)
# 
# logit_confusion_matrix_amount_smaller_10 <- ggplot( result, aes( actual, predict, color=amount_smaller_10) ) + 
#   geom_violin( fill = "white", color = NA ) +
#   geom_jitter( shape = 1, size= 2.5) + 
#   scale_y_continuous( limits = c( 0, 1 ) ) + 
#   scale_color_manual(name="Amount < 10", values=c("black","blue"))+
#   guides( col = guide_legend( nrow = 2 ) ) +#+ # adjust the legend to have two rows  
#   ggtitle( sprintf( "CI") ) +
#   labs(y = "P(Fraudulent)")+
#   theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
#         panel.background = element_blank(), axis.line = element_line(colour = "black"))
# 
# 
# ## ROC CDCS
# results_class<- dplyr::filter(results, Processing=="class")
# median_method_AUC <- dplyr::filter(results_class, AUC==AUC[which.min(abs((median(results_class$AUC)-0.00001)-AUC))])
# median_Processing <- unique(median_method_AUC$Processing)
# median_k_run <- unique(median_method_AUC$k_run)
# median_j_fold <- unique(median_method_AUC$j_fold)
# 
# threshold_BMR_class <- dplyr::filter(threshold_BMR, k_run==median_k_run, j_fold==median_j_fold )
# 
# AUC_class <- round(unique(median_method_AUC$AUC),3)
# 
# testset_with_prediction_class <- dplyr::filter(testset_with_prediction_methods, Processing==median_Processing, k_run==median_k_run, j_fold==median_j_fold)
# testset_with_prediction_class
# 
# tpr_fpr_precision_recall_class <- select(median_method_AUC, TPR, FPR, Precision, Recall)
# tpr_fpr_precision_recall_class$Savings <- median_method_AUC$Savings
# thresholds_names <- c("0.5","Imbalance ratio","Class cost", "Instance cost", "Youden optimal", "F1 optimal", "Savings optimal")
# tpr_fpr_precision_recall_class$thresholds <- thresholds_names
# 
# tpr_fpr_class <-select(tpr_fpr_precision_recall_class,TPR, FPR, Savings, thresholds)
# precision_recall_class <- select(tpr_fpr_precision_recall_class, Precision, Recall, Savings, thresholds)
# 
# tpr_fpr_overlap_class <- dplyr::filter(tpr_fpr_class, round(TPR,3)==0.828)
# tpr_fpr_no_overlap_class <- dplyr::filter(tpr_fpr_class, round(TPR,3)!=0.828)
# 
# pr <- prediction(testset_with_prediction_class$scores, testset_with_prediction_class$Class) #Ok because we haven't specified the threshold, the ROC curve is drawn for all thresholds
# perf <- ROCR::performance(pr, measure = "tpr", x.measure = "fpr") #Ok the performance function is used to draw the ROC curve
# roc_dt <- data.frame( fpr = perf@x.values[[1]], tpr = perf@y.values[[1]] )
# 
# class_roc <- ggplot( roc_dt, aes( fpr, tpr )) + 
#   geom_line( color = rgb( 0, 0, 1, alpha = 0.8 ) ) +
#   geom_label_repel( data=tpr_fpr_no_overlap_instance, mapping=aes(x=FPR, y= TPR, label = thresholds),
#                     box.padding   = 0.35, 
#                     point.padding = 0.4,
#                     nudge_x = 0.01,
#                     nudge_y = -0.15,
#                     segment.color = 'grey50') +
#   labs( title = paste("ROC: CDCS", "( AUC =",AUC_class,")"), x = "False Postive Rate", y = "True Positive Rate" )+
#   geom_point(data=tpr_fpr_class, aes(x=FPR, y= TPR,color=Savings), size = 2, alpha = 0.8 ) +
#   scale_color_gradient(low="red", high="green")+
#   theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
#         panel.background = element_blank(), axis.line = element_line(colour = "black"))
# 
# arrange(select(dplyr::filter(testset_with_prediction_class, Class==1, scores< 0.8), scores, Class), desc(scores))
# arrange(select(dplyr::filter(testset_with_prediction_class, Class==0, scores< 1), scores, Class), desc(scores))
# 
# 
# ## ROC CDCS ZOOM
# class_roc_zoom <- ggplot( roc_dt, aes( fpr, tpr )) + 
#   geom_line( color = rgb( 0, 0, 1, alpha = 0.8 ) ) +
#   labs( title = "ROC: cslogit (class)", x = "False Postive Rate", y = "True Positive Rate" )+
#   geom_point(  data=tpr_fpr_overlap_class, aes(x=FPR, y= TPR, color=Savings), size = 2, alpha = 0.8) +
#   geom_label_repel( data=tpr_fpr_overlap_class, mapping=aes(x=FPR, y= TPR, label = thresholds),
#                     box.padding   = 0.6, 
#                     point.padding = 0.5,
#                     nudge_x = 0,
#                     nudge_y = -0.003,
#                     force=10,
#                     force_pull=0,
#                     segment.color = 'grey50') +
#   scale_color_gradient(low="red", high="green")+
#   xlim(-0.0,0.005)+
#   ylim(0.815, 0.845)+
#   theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
#         panel.background = element_blank(), axis.line = element_line(colour = "black"))
# 
# 
# ## Precision-recall CDCS
# perf <- ROCR::performance(pr, measure="prec", x.measure="rec") 
# precision_recall_dt <- data.frame( rec = perf@x.values[[1]], prec = perf@y.values[[1]] )
# 
# class_precision_recall <- ggplot( precision_recall_dt, aes( rec, prec ), label=Name) + 
#   geom_line( color = rgb( 0, 0, 1, alpha = 0.8 ) ) +
#   geom_point( data=precision_recall_class, aes(x=Recall, y= Precision, color=Savings), size = 2, alpha = 0.8 ) +
#   labs( title = paste("Precision-Recall: CDCS", "( AUC =",AUC_class,")"), x = "Recall", y = "Precision", col="Savings" )+
#   geom_label_repel(data=precision_recall_class, mapping=aes(x=Recall, y= Precision, label = thresholds),
#                    box.padding   = 0.45, 
#                    point.padding = 0.5,
#                    nudge_x = -0.32,
#                    nudge_y = -0.05,
#                    force = 1.2,
#                    segment.color = 'grey50' ) +
#   scale_color_gradient(low="red", high="green")+
#   theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
#         panel.background = element_blank(), axis.line = element_line(colour = "black"))
# 
# 
# ## Confusion Matrixes CDCS
# TP_FP_TN_FN <- select(median_method_AUC, TP, FP, TN, FN)
# TP_FP_TN_FN$thresholds <- thresholds_names
# 
# threshold_BMR_class <- dplyr::filter(threshold_BMR, k_run==median_k_run, j_fold==median_j_fold )
# 
# cutoff <- 0.5
# 
# predict <- testset_with_prediction_class$scores
# actual  <- relevel(as.factor(testset_with_prediction_class$Class), "1")
# 
# result <- cbind.data.frame( actual = actual, predict = predict )
# 
# result$type <- ifelse( predict >= cutoff & actual == 1, "TP",
#                        ifelse( predict >= cutoff & actual == 0, "FP", 
#                                ifelse( predict <  cutoff & actual == 1, "FN", "TN" ) ) )
# result$type <- as.factor(result$type)
# 
# class_confusion_matrix_constant <- ggplot( result, aes( actual, predict) ) + 
#   geom_violin( fill = "white", color = NA ) +
#   geom_jitter( shape = 1, size= 2.5) + 
#   scale_y_continuous( limits = c( 0, 1 ) ) + 
#   #scale_color_discrete( breaks = c( "TP", "FN", "FP", "TN" ) ) + # ordering of the legend 
#   guides( col = guide_legend( nrow = 2 ) ) +#+ # adjust the legend to have two rows  
#   ggtitle( sprintf( "CDCS") ) +
#   labs(y = "P(Fraudulent)")+
#   theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
#         panel.background = element_blank(), axis.line = element_line(colour = "black"))
# 
# class_confusion_matrix_0.5 <- ggplot( result, aes( actual, predict, color = type ) ) + 
#   geom_violin( fill = "white", color = NA ) +
#   geom_jitter( shape = 1, size= 2.5) + 
#   geom_hline( yintercept = cutoff, color = "blue", alpha = 0.6 ) + 
#   scale_y_continuous( limits = c( 0, 1 ) ) + 
#   scale_color_discrete( breaks = c( "TP", "FN", "FP", "TN" ) ) + # ordering of the legend 
#   guides( col = guide_legend( nrow = 2 ) ) +#+ # adjust the legend to have two rows  
#   ggtitle( sprintf( "Confusion Matrix: CDCS with 0.5 threshold") ) +
#   labs(y = "P(Fraudulent)")+
#   theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
#         panel.background = element_blank(), axis.line = element_line(colour = "black"))
# 
# cutoff <- threshold_BMR_class$threshold_instance_cost
# 
# predict <- testset_with_prediction_class$scores
# actual  <- relevel(as.factor(testset_with_prediction_class$Class), "1")
# 
# result <- cbind.data.frame( actual = actual, predict = predict )
# 
# result$type <- ifelse( predict >= cutoff & actual == 1, "TP",
#                        ifelse( predict >= cutoff & actual == 0, "FP", 
#                                ifelse( predict <  cutoff & actual == 1, "FN", "TN" ) ) )
# result$type <- as.factor(result$type)
# 
# class_confusion_matrix_instance <- ggplot( result, aes( actual, predict, color = type ) ) + 
#   geom_violin( fill = "white", color = NA ) +
#   geom_jitter( shape = 1, size= 2.5) + 
#   scale_y_continuous( limits = c( 0, 1 ) ) + 
#   scale_color_discrete( breaks = c( "TP", "FN", "FP", "TN" ) ) + # ordering of the legend 
#   guides( col = guide_legend( nrow = 2 ) ) +#+ # adjust the legend to have two rows  
#   ggtitle( sprintf( "Confusion Matrix: CDCS with Instance cost threshold") ) +
#   labs(y = "P(Fraudulent)")+
#   theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
#         panel.background = element_blank(), axis.line = element_line(colour = "black"))
# 
# amount_test_cv_class <- dplyr::filter(amount_test_cv, k_run==median_k_run, j_fold==median_j_fold)
# amount_test_cv_class <- select(amount_test_cv_class, amount_test)
# 
# result$amount_smaller_10 <- ifelse(amount_test_cv_class<10, "Yes", "No")
# result$amount_smaller_10 <- as.factor(result$amount_smaller_10)
# 
# class_confusion_matrix_amount_smaller_10 <- ggplot( result, aes( actual, predict, color=amount_smaller_10) ) + 
#   geom_violin( fill = "white", color = NA ) +
#   geom_jitter( shape = 1, size= 2.5) + 
#   scale_y_continuous( limits = c( 0, 1 ) ) + 
#   scale_color_manual(name="Amount < 10", values=c("black","blue"))+
#   guides( col = guide_legend( nrow = 2 ) ) +#+ # adjust the legend to have two rows  
#   ggtitle( sprintf( "CDCS") ) +
#   labs(y = "P(Fraudulent)")+
#   theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
#         panel.background = element_blank(), axis.line = element_line(colour = "black"))
# 
# 
# ## ROC IDCS
# results_instance<- dplyr::filter(results, Processing=="instance")
# median_method_AUC <- dplyr::filter(results_instance, AUC==AUC[which.min(abs((median(results_instance$AUC)-0.00001)-AUC))])
# median_Processing <- unique(median_method_AUC$Processing)
# median_k_run <- unique(median_method_AUC$k_run)
# median_j_fold <- unique(median_method_AUC$j_fold)
# 
# AUC_instance <- round(unique(median_method_AUC$AUC),3)
# 
# testset_with_prediction_instance <- dplyr::filter(testset_with_prediction_methods, Processing==median_Processing, k_run==median_k_run, j_fold==median_j_fold)
# testset_with_prediction_instance
# 
# tpr_fpr_precision_recall_instance <- select(median_method_AUC, TPR, FPR, Precision, Recall)
# tpr_fpr_precision_recall_instance$Savings <- median_method_AUC$Savings
# thresholds_names <- c("0.5","Imbalance ratio","Class cost", "Instance cost", "Youden optimal", "F1 optimal", "Savings optimal")
# tpr_fpr_precision_recall_instance$thresholds <- thresholds_names
# 
# tpr_fpr_instance <-select(tpr_fpr_precision_recall_instance,TPR, FPR, Savings, thresholds)
# precision_recall_instance <- select(tpr_fpr_precision_recall_instance, Precision, Recall, Savings, thresholds)
# 
# tpr_fpr_overlap_instance <- dplyr::filter(tpr_fpr_instance, round(TPR,3)==0.802)
# tpr_fpr_no_overlap_instance <- dplyr::filter(tpr_fpr_instance, round(TPR,3)!=0.802)
# 
# pr <- prediction(testset_with_prediction_instance$scores, testset_with_prediction_instance$Class) #Ok because we haven't specified the threshold, the ROC curve is drawn for all thresholds
# perf <- ROCR::performance(pr, measure = "tpr", x.measure = "fpr") #Ok the performance function is used to draw the ROC curve
# roc_dt <- data.frame( fpr = perf@x.values[[1]], tpr = perf@y.values[[1]] )
# 
# instance_roc <- ggplot( roc_dt, aes( fpr, tpr ), label=Name) + 
#   geom_line( color = rgb( 0, 0, 1, alpha = 0.8 ) ) +
#   labs( title = paste("ROC: IDCS", "( AUC = ",AUC_instance,")"), x = "False Postive Rate", y = "True Positive Rate" )+
#   geom_point(data=tpr_fpr_instance, aes(x=FPR, y= TPR, color=Savings), size = 2, alpha = 0.8) +
#   geom_label_repel( data=tpr_fpr_no_overlap_instance, mapping=aes(x=FPR, y= TPR, label = thresholds),
#                     box.padding   = 0.35, 
#                     point.padding = 0.4,
#                     nudge_x = 0.01,
#                     nudge_y = -0.15,
#                     segment.color = 'grey50') +
#   scale_color_gradient(low="red", high="green")+
#   theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
#         panel.background = element_blank(), axis.line = element_line(colour = "black"))
# 
# arrange(select(dplyr::filter(testset_with_prediction_instance, Class==1, scores< 1), scores, Class), desc(scores))
# arrange(select(dplyr::filter(testset_with_prediction_instance, Class==0, scores< 1), scores, Class), desc(scores))
# 
# 
# ## Roc IDCS ZOOM
# instance_roc_zoom <- ggplot( roc_dt, aes( fpr, tpr )) + 
#   geom_line( color = rgb( 0, 0, 1, alpha = 0.8 ) ) +
#   labs( title = paste("ROC: IDCS", "( AUC = ",AUC_instance,")"), x = "False Postive Rate", y = "True Positive Rate" )+
#   geom_point(  data=tpr_fpr_overlap_instance, aes(x=FPR, y= TPR, color=Savings), size = 2, alpha = 0.8) +
#   geom_label_repel( data=tpr_fpr_overlap_instance, mapping=aes(x=FPR, y= TPR, label = thresholds),
#                     box.padding   = 0.6, 
#                     point.padding = 0.5,
#                     nudge_x = 0.0001,
#                     nudge_y = -0.001,
#                     force=20,
#                     force_pull =0,
#                     segment.color = 'grey50') +
#   scale_color_gradient(low="red", high="green")+
#   xlim(0, 0.003)+
#   ylim(0.80, 0.8050)+
#   theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
#         panel.background = element_blank(), axis.line = element_line(colour = "black"))
# 
# 
# ## Precision-recall IDCS
# perf <- ROCR::performance(pr, measure="prec", x.measure="rec") 
# precision_recall_dt <- data.frame( rec = perf@x.values[[1]], prec = perf@y.values[[1]] )
# instance_precision_recall <- ggplot( precision_recall_dt, aes( rec, prec ), label=Name) + 
#   geom_line( color = rgb( 0, 0, 1, alpha = 0.8 ) ) +
#   geom_point( data=precision_recall_instance, aes(x=Recall, y= Precision, color=Savings), size = 2, alpha = 0.8 ) +
#   labs( title = paste("Precision-Recall: IDCS", "( AUC =",AUC_instance,")"), x = "Recall", y = "Precision", col="Savings" )+
#   geom_label_repel(data=precision_recall_instance, mapping=aes(x=Recall, y= Precision, label = thresholds),
#                    box.padding   = 0.35, 
#                    point.padding = 0.5,
#                    nudge_x = -0.32,
#                    nudge_y = -0.05,
#                    force = 1.2,
#                    segment.color = 'grey50' ) +
#   xlim(-0.4,1)+
#   scale_color_gradient(low="red", high="green")+
#   theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
#         panel.background = element_blank(), axis.line = element_line(colour = "black")) 
# 
# ## Confusion Matrixes IDCS
# TP_FP_TN_FN <- select(median_method_AUC, TP, FP, TN, FN)
# TP_FP_TN_FN$thresholds <- thresholds_names
# 
# threshold_BMR_instance <- dplyr::filter(threshold_BMR, k_run==median_k_run, j_fold==median_j_fold )
# 
# cutoff <- 0.5
# 
# predict <- testset_with_prediction_instance$scores
# actual  <- relevel(as.factor(testset_with_prediction_instance$Class), "1")
# 
# result <- cbind.data.frame( actual = actual, predict = predict )
# 
# result$type <- ifelse( predict >= cutoff & actual == 1, "TP",
#                        ifelse( predict >= cutoff & actual == 0, "FP", 
#                                ifelse( predict <  cutoff & actual == 1, "FN", "TN" ) ) )
# result$type <- as.factor(result$type)
# 
# instance_confusion_matrix_constant <- ggplot( result, aes( actual, predict) ) + 
#   geom_violin( fill = "white", color = NA ) +
#   geom_jitter( shape = 1, size= 2.5) + 
#   scale_y_continuous( limits = c( 0, 1 ) ) + 
#   #scale_color_discrete( breaks = c( "TP", "FN", "FP", "TN" ) ) + # ordering of the legend 
#   guides( col = guide_legend( nrow = 2 ) ) +#+ # adjust the legend to have two rows  
#   ggtitle( sprintf( "IDCS", cutoff ) ) +
#   labs(y = "P(Fraudulent)")+
#   theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
#         panel.background = element_blank(), axis.line = element_line(colour = "black"))
# 
# instance_confusion_matrix_0.5 <- ggplot( result, aes( actual, predict, color = type ) ) + 
#   geom_violin( fill = "white", color = NA ) +
#   geom_jitter( shape = 1, size= 2.5) + 
#   scale_y_continuous( limits = c( 0, 1 ) ) + 
#   geom_hline( yintercept = cutoff, color = "blue", alpha = 0.6 ) + 
#   scale_color_discrete( breaks = c( "TP", "FN", "FP", "TN" ) ) + # ordering of the legend 
#   guides( col = guide_legend( nrow = 2 ) ) +#+ # adjust the legend to have two rows  
#   ggtitle( sprintf( "Confusion Matrix: IDCS with 0.5 threshold") ) +
#   labs(y = "P(Fraudulent)")+
#   theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
#         panel.background = element_blank(), axis.line = element_line(colour = "black"))
# 
# cutoff <- threshold_BMR_instance$threshold_instance_cost
# 
# predict <- testset_with_prediction_instance$scores
# actual  <- relevel(as.factor(testset_with_prediction_instance$Class), "1")
# 
# result <- cbind.data.frame( actual = actual, predict = predict )
# 
# result$type <- ifelse( predict >= cutoff & actual == 1, "TP",
#                        ifelse( predict >= cutoff & actual == 0, "FP", 
#                                ifelse( predict <  cutoff & actual == 1, "FN", "TN" ) ) )
# result$type <- as.factor(result$type)
# 
# instance_confusion_matrix_instance <- ggplot( result, aes( actual, predict, color = type ) ) + 
#   geom_violin( fill = "white", color = NA ) +
#   geom_jitter( shape = 1, size= 2.5) + 
#   scale_y_continuous( limits = c( 0, 1 ) ) + 
#   scale_color_discrete( breaks = c( "TP", "FN", "FP", "TN" ) ) + # ordering of the legend 
#   guides( col = guide_legend( nrow = 2 ) ) +#+ # adjust the legend to have two rows  
#   ggtitle( sprintf( "Confusion Matrix: IDCS with Instance cost threshold") ) +
#   labs(y = "P(Fraudulent)")+
#   theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
#         panel.background = element_blank(), axis.line = element_line(colour = "black"))
# 
# amount_test_cv_instance <- dplyr::filter(amount_test_cv, k_run==median_k_run, j_fold==median_j_fold)
# amount_test_cv_instance <- select(amount_test_cv_instance, amount_test)
# 
# result$amount_smaller_10 <- ifelse(amount_test_cv_instance<10, "Yes", "No")
# result$amount_smaller_10 <- as.factor(result$amount_smaller_10)
# 
# instance_confusion_matrix_amount_smaller_10 <- ggplot( result, aes( actual, predict, color=amount_smaller_10) ) + 
#   geom_violin( fill = "white", color = NA ) +
#   geom_jitter( shape = 1, size= 2.5) + 
#   scale_y_continuous( limits = c( 0, 1 ) ) + 
#   scale_color_manual(name="Amount < 10", values=c("black","blue"))+
#   guides( col = guide_legend( nrow = 2 ) ) +#+ # adjust the legend to have two rows  
#   ggtitle( sprintf( "IDCS") ) +
#   labs(y = "P(Fraudulent)")+
#   theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
#         panel.background = element_blank(), axis.line = element_line(colour = "black"))
# 
# 
# 
# 
# 
# if(nruns == 25){
#   ### Coefficients on 25 runs ----------------------------------------------------
#   ## Intercept
#   Intercept_logit <- data_frame(coef_logit$X.Intercept.)
#   Intercept_logit$method <- c(rep("CI", 50))
#   Intercept_class <- data_frame(coef_class$X.Intercept.)
#   Intercept_class$method <- c(rep("CDCS", 50))
#   Intercept_instance <- data.frame(coef_instance$X.Intercept.)
#   Intercept_instance$method <- c(rep("IDCS", 50))
#   
#   colnames(Intercept_logit)[1] <- "Intercept"
#   colnames(Intercept_class)[1] <- "Intercept"
#   colnames(Intercept_instance)[1] <- "Intercept"
#   
#   Intercept_methods <- rbind.data.frame(Intercept_logit, Intercept_class, Intercept_instance)
#   
#   Intercept_methods %>% 
#     group_by(method) %>% 
#     summarise(Intercept_avg = mean(Intercept), Intercept_std = sd(Intercept)) %>% 
#     arrange(factor(method, levels=c("CI","CDCS","IDCS")))
#   #or
#   tapply(Intercept_methods$Intercept, Intercept_methods$method, mean)
#   
#   Intercept_boxplot <- ggplot( data = Intercept_methods, mapping = aes(x = method, y = Intercept, fill = method)) +
#     stat_boxplot(geom = "errorbar", width = 0.4) + geom_boxplot() + 
#     xlab("") + scale_x_discrete(limits = unique(Intercept_methods$method)) +
#     stat_summary(fun = mean, geom = "point", shape = 18, size = 5, col = "black")+
#     theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
#           panel.background = element_blank(), axis.line = element_line(colour = "black"))+
#     theme(text = element_text(size = 15), legend.position = "none", axis.text.x = element_text(angle = 0))
#   
#   
#   ## LogAmount
#   LogAmount_logit <- data_frame(coef_logit$LogAmount)
#   LogAmount_logit$method <- c(rep("CI", 50))
#   LogAmount_class <- data_frame(coef_class$LogAmount)
#   LogAmount_class$method <- c(rep("CDCS", 50))
#   LogAmount_instance <- data.frame(coef_instance$LogAmount)
#   LogAmount_instance$method <- c(rep("IDCS", 50))
#   
#   colnames(LogAmount_logit)[1] <- "LogAmount"
#   colnames(LogAmount_class)[1] <- "LogAmount"
#   colnames(LogAmount_instance)[1] <- "LogAmount"
#   
#   LogAmount_methods <- rbind.data.frame(LogAmount_logit, LogAmount_class, LogAmount_instance)
#   
#   LogAmount_methods %>% 
#     group_by(method) %>% 
#     summarise(LogAmounts_avg = mean(LogAmount),LogAmounts_std = sd(LogAmount)) %>% 
#     arrange(factor(method, levels=c("CI","CDCS","IDCS")))
#   #or
#   tapply(LogAmount_methods$LogAmount, LogAmount_methods$method, mean)
#   
#   LogAmount_boxplot <- ggplot( data = LogAmount_methods, mapping = aes(x = method, y = LogAmount, fill = method)) +
#     stat_boxplot(geom = "errorbar", width = 0.4) + geom_boxplot() + ylab("LogAmount coefficient")+
#     xlab("") + scale_x_discrete(limits = unique(LogAmount_methods$method)) +
#     stat_summary(fun = mean, geom = "point", shape = 18, size = 5, col = "black")+
#     theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
#           panel.background = element_blank(), axis.line = element_line(colour = "black"))+
#     theme(text = element_text(size = 15), legend.position = "none", axis.text.x = element_text(angle = 0))
# }
# 
# 
# 
# 
# 
# ### Figures---------------------------------------------------------------------
# box_savings_best
# 
# average_0.5_boxplot_savings
# average_0.5_boxplot_AUC
# average_0.5_boxplot_F1
# 
# all_results_boxplot_Savings
# all_results_boxplot_F1
# all_results_boxplot_AUC
# 
# logit_roc
# logit_precision_recall
# logit_confusion_matrix_constant
# logit_confusion_matrix_amount_smaller_10
# logit_confusion_matrix_0.5
# logit_confusion_matrix_instance
# 
# class_roc
# class_roc_zoom
# class_precision_recall
# class_confusion_matrix_constant
# class_confusion_matrix_amount_smaller_10
# class_confusion_matrix_0.5
# class_confusion_matrix_instance
# 
# instance_roc
# instance_roc_zoom
# instance_precision_recall
# instance_confusion_matrix_constant
# instance_confusion_matrix_amount_smaller_10
# instance_confusion_matrix_0.5
# instance_confusion_matrix_instance
# 
# ## nruns = 25
# Intercept_boxplot
# LogAmount_boxplot

 





