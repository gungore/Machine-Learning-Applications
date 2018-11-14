library(xgboost)
library(data.table)
library(dplyr)
library(tidyr)
setwd("C:/Users/trgungoe/Desktop/project instacart")

train<- fread("items_train_setv1newfeatureskmeans.csv") 
test <- fread("items_test_setv1newfeaturekmeans.csv")

# Train / Test datasets --------------------------------------------------- 
train$user_id <- NULL
train$product_id <- NULL
train$reordered_x[is.na(train$reordered_x)] <- 0

test$user_id <- NULL
test$reordered_x <- NULL
 
params <- list(
  "objective"           = "reg:logistic",
  "eval_metric"         = "logloss",
  "eta"                 = 0.1,#0.09  #ilk halleri
  "max_depth"           = 7, #
  "min_child_weight"    = 9, #10
  "gamma"               = 0.80, #0.70
  "subsample"           = 0.76, #0.76
  "colsample_bytree"    = 0.95,
  "alpha"               = 0,
  "lambda"              = 1
)

subtrain <- train %>% sample_frac(0.4)
X <- xgb.DMatrix(as.matrix(subtrain %>% select(-reordered_x)), label = subtrain$reordered_x)
model <- xgboost(data = X, params = params, nrounds = 90)

importance <- xgb.importance(colnames(X), model = model)
xgb.ggplot.importance(importance)

rm(X, importance, subtrain)
gc()
 

# Apply model -------------------------------------------------------------
X <- xgb.DMatrix(as.matrix(test %>% select(-order_id, -product_id)))
test$reordered_x <- predict(model, X)

test$reordered_x <- (test$reordered_x > 0.20) * 1 


submission <- test %>%
  filter(reordered_x == 1) %>%
  group_by(order_id) %>%
  summarise(
    products = paste(product_id, collapse = " ")
  )

missing <- data.frame(
  order_id = unique(test$order_id[!test$order_id %in% submission$order_id]),
  products = "None"
)
submission <- submission %>% bind_rows(missing) %>% arrange(order_id)
write.csv(submission, file = "submitfinalv1newfeatures19092017.csv", row.names = F)


#Cross-Validation

params <- list(
  "objective"           = "reg:logistic",
  "eval_metric"         = "auc",
  "eta"                 = 0.1,
  "max_depth"           = 7, 
  "min_child_weight"    = 9, 
  "gamma"               = 0.80, 
  "subsample"           = 0.76, 
  "colsample_bytree"    = 0.95,
  "alpha"               = 0,
  "lambda"              = 1
)
nround    <- 10 # number of XGBoost rounds
cv.nfold  <- 5
cv_model <- xgb.cv(params =params,
                   data = X, 
                   nrounds = nround,
                   nfold = cv.nfold,
                   verbose = FALSE,
                   prediction = TRUE)



