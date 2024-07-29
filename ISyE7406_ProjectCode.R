#### decided to use binary classifier instead ###

library(tidymodels)
library(randomForest)
library(data.table)
library(caret)
library(MASS)
library(naivebayes)
library(neuralnet)
library(class)
library(car)
library(corrplot)
library(dplyr)
library(gbm)
library(stats)
library(pROC)
library(xgboost)
set.seed(123)


######################################################################################################
### changed data set to balanced data set
diabetes <- read.table(file= "C:/Users/rkang/OneDrive/Documents/OMSA/2024-01_Spring/ISYE 7406 - DMSL/project/diabetes/diabetes_binary_5050split_health_indicators_BRFSS2015.csv", header=TRUE, sep=",")
dim(diabetes)

diabetes = as.data.frame(lapply(diabetes, as.numeric))
# create a sample of the data set bc it is too large to work with on my computer
# if we were to want to actually use this modeling in real life, would use all data with a more powerful computer
set.seed(123)
sample_idx <- createDataPartition(diabetes$Diabetes_binary, p = .2, list=FALSE)
sample_diabetes <- diabetes[sample_idx,]
data_len <- dim(sample_diabetes)[1]
sample_diabetes <- as.data.frame(apply(sample_diabetes, 2, function(x) (x-min(x))/(max(x)-min(x))))
summary(sample_diabetes)
######################################################################################################





######################################################################################################
## Split to training and testing subset 
training_count = round(data_len*.7)
testing_count = data_len - training_count

flag <- sort(sample(data_len, testing_count, replace = FALSE))
diabetestrain <- sample_diabetes[-flag,]
diabetestest <- sample_diabetes[flag,]

## Extract the true response value for training and testing data
y1 <- diabetestrain$Diabetes_binary;
y2 <- diabetestest$Diabetes_binary;
######################################################################################################





######################################################################################################
# variable selection for non-ensemble methods:
set.seed(123)
full_model <- glm(as.factor(Diabetes_binary) ~., family = binomial(link="logit"), data=diabetestrain)
model_select_variables  <- stats::step(full_model, direction = "both");
## see coefficients of model_select_variables
round(coef(model_select_variables),3)
summary(model_select_variables)
# best model according to AIC is HighBP + HighChol + BMI + HeartDiseaseorAttack + GenHlth + MentHlth + Age
# best model according to AIC with 20% of balanced data set is HighBP + HighChol + CholCheck + BMI + Stroke + HeartDiseaseorAttack + Veggies + HvyAlcoholConsump + AnyHealthcare + NoDocbcCost + GenHlth + PhysHlth + Sex + Age + Income 

# vif to check for multicollinearity
vif_results <- vif(full_model)
print(vif_results)
# None have high VIF values
# At 20% of balanced, even better - all lower than 2
######################################################################################################






######################################################################################################
#EDA
#### (c)
# summary
summary(sample_diabetes)

# histograms of top 4 variables chosen by feature selection
hist(sample_diabetes$Diabetes_binary, main ="Histogram of Diabetes_binary", xlab="Diabetes_binary")
hist(sample_diabetes$HighBP, main ="Histogram of HighBP", xlab="HighBP")
hist(sample_diabetes$HighChol, main ="Histogram of HighChol", xlab="HighChol")
hist(sample_diabetes$CholCheck, main ="Histogram of CholCheck", xlab="CholCheck")
hist(sample_diabetes$BMI, main ="Histogram of BMI", xlab="BMI")

# boxplots of top 4 variables chosen by feature selection
boxplot(sample_diabetes$Diabetes_binary, main ="Boxplot of Diabetes_binary", xlab="Diabetes_binary")
boxplot(sample_diabetes$HighBP, main ="Boxplot of HighBP", xlab="HighBP")
boxplot(sample_diabetes$HighChol, main ="Boxplot of HighChol", xlab="HighChol")
boxplot(sample_diabetes$CholCheck, main ="Boxplot of CholCheck", xlab="CholCheck")
boxplot(sample_diabetes$BMI, main ="Boxplot of BMI", xlab="BMI")

# correlation
corr_data <- cor(sample_diabetes)
corrplot(corr_data, method = "color", order="hclust", main = "\n Correlation Between Variables in Diabetes Data Set")

corr_data_filtered <- cor(sample_diabetes)
corr_data_filtered[abs(corr_data_filtered) <= .3 | corr_data_filtered == 1] <- 0
corrplot(corr_data_filtered, method = "color", order="hclust", main = "\n Correlation Between Variables in Diabetes Data Set (Filtered for Absolute Value > 0.3)")
######################################################################################################







######################################################################################################
## Build Random Forest with the default parameters
## It can be 'classification', 'regression', or 'unsupervised'
# the default for randomForest is classification
# default ntree is 500, default mtry is square root of number of predictors, default nodesize is 1
# rf's automatically perform feature selection
set.seed(123)
rf1 <- randomForest(as.factor(Diabetes_binary) ~., data=diabetestrain, importance=TRUE)

## Check Important variables; the larger the value, the more important the variable
importance(rf1)
## There are two types of importance measure 
##  (1=mean decrease in accuracy, 2= mean decrease in node impurity); classification, only need to consider type 2
importance(rf1, type=2)
# will show you which variables are important - the top ones are most important
varImpPlot(rf1)

## Prediction on the testing data set
# type = 'class' if doing classification
rf.pred = predict(rf1, diabetestest[,-1], type='class')
# misclassification rate
table(rf.pred, y2)
acc = mean(rf.pred == y2)
te <- 1 - acc # 0.1023622
# node size 1 can lead to overfitting


##In practice, You can fine-tune parameters in Random Forest such as 
#ntree = number of tress to grow, and the default is 500. 
#mtry = number of variables randomly sampled as candidates at each split. 
#          The default is sqrt(p) for classfication and p/3 for regression
#nodesize = minimum size of terminal nodes. 
#           The default value is 1 for classification and 5 for regression

# can try doing this isn for loop with diff values of ntree, mtry, and nodesize to identify
# the right parameters that minimize cross-validation errors
# Bc I have smaller subset of data, I'm using ntree values between 100 to 500
# nodesize --- loop between 3 through 6 --- skip 1 and 2 due to overfitting

## In general, we need to use a loop to try different parameter
## values (of ntree, mtry, etc.) to identify the right parameters 
## that minimize cross-validation errors.
# cross validating on training set to tune parameters
# I had a hard time getting packages to work when doing parameter tuning, so I just used a nested loop

ntrees <- c(100,200,300,400,500)
mtrys <- c(1) # used to have as 5, 6, 7 but it was causing overfitting. mtry of 1 is good for high dimensional data set
nodesizes <- c(3,4,5,6,7)

min_cverror <- 1.0
for (ntree in ntrees) {
  for (mtry in mtrys) {
    for (nodesize in nodesizes) {
      cverror <- NULL;
      set.seed(123)
      five_folds <- createFolds(diabetestrain$Diabetes_binary, k = 5, list=TRUE)
      for (i in 1:5) {
        set.seed(123)
        train <- diabetestrain[unlist(five_folds[-i]),]
        validation <- diabetestrain[unlist(five_folds[i]),]
        set.seed(123)
        rf2 <- randomForest(as.factor(Diabetes_binary) ~., data=train, ntree=ntree, 
                            mtry=mtry, nodesize=nodesize, importance=TRUE)
        rf.pred2 <- predict(rf2, validation, type='class')
        acc <- mean(rf.pred2 == validation$Diabetes_binary)
        te <- 1-acc
        cverror <- c(cverror, te)
      }
      mean_cverror <- mean(cverror)
      if (mean_cverror < min_cverror) {
        min_cverror <- mean_cverror
        best_params <- c(paste("ntree =",as.character(ntree),", mtry =",as.character(mtry), ", nodesize =", as.character(nodesize)))
      }
    }
  }
}

min_cverror # 0.1498127
best_params # best params are ntree=500, mtry=5, and nodesize=7
# best params for 5% are: "ntree = 100 , mtry = 5 , nodesize = 6"
# best params for 20% of balanced are: ntree=500, mtry=5, and nodesize=7
# best params for 20% of balanced are: "ntree = 100 , mtry = 1 , nodesize = 4" when setting mtry to 1 since we don't want overfitting

# best params are ntree=500, mtry=5, and nodesize=7, so use these to fit a new random forest model
set.seed(123)
rf3 <- randomForest(as.factor(Diabetes_binary) ~., data=diabetestrain, ntree = 100 , mtry = 1 , nodesize = 4, importance=TRUE)
importance(rf3)
importance(rf3, type=2)
varImpPlot(rf3)

## Prediction on the testing data set
# type = 'class' if doing classification
rf3.pred = predict(rf3, diabetestest[,-1], type='class')
# misclassification rate
table(rf3.pred, y2)
acc3 = mean(rf3.pred == y2)
te3 <- 1 - acc3 # 0.1128609

# rf.pred_tr = predict(rf3, validation[,-1], type='class')
# acc_tr = mean(rf.pred_tr == validation$Diabetes_binary)
# tre <- 1 - acc_tr # 0.1023622
######################################################################################################





######################################################################################################
# BOOSTING
# right now, I have as default values n.trees = 100, shrinkage = 0.1, interaction.depth = 1
set.seed(123)
gbm.diabetes1 <- gbm(Diabetes_binary~.,data=diabetestrain,
                 distribution = 'bernoulli',
                 n.trees = 100, 
                 shrinkage = 0.1, 
                 interaction.depth = 1,
                 cv.folds = 5)

## Model Inspection 
## Find the estimated optimal number of iterations
# to find the optimal M, use:
# blue line gives optimal number of M trees or iterations from CV. 
# the black and green lines are deviance (-2*log-likelihood) of training (black) and testing (green) data
perf_gbm1 = gbm.perf(gbm.diabetes1, method="cv") 
perf_gbm1

# you can compute the relative influence of each X variable in the gbm object using summary
# this is similar to the variable importance measures for random forests
# except using the entire training dataset (not the out-of-bag observations)
summary(gbm.diabetes1)

# The n.trees value can be the optimal value from the model inspection
# since I got the optimal value being 100:
# since type = 'response', need to change back to class labels (the default of predict fxn is to return the log-odd-ratio values)
# when we set type = 'response', it returns probabilities, and we need to manually change back to class labels or 
# else you'll get lot of errors
## Training error
pred1gbm <- predict(gbm.diabetes1, newdata = diabetestrain[,-1], n.trees=perf_gbm1, type="response")
pred1gbm[1:10]

y1hat <- ifelse(pred1gbm < 0.5, 0, 1)
y1hat[1:10]
sum(y1hat != y1)/length(y1)  ##Training error = 0.1463964

## Testing Error
y2hat <- ifelse(predict(gbm.diabetes1,newdata = diabetestest[,-1], n.trees=perf_gbm1, type="response") < 0.5, 0, 1)
mean(y2hat != y2) 
## Testing error = 0.1076115


####now loop through multiple values of the parameters to see which is best
# build the boosting model from the training data
shrinkages <- c(0.05, 0.1, 0.15, 0.2)
depths <- c(1, 2, 3, 4, 5)
cverror2 = NULL;
best_gbm_te <- 1.0

# doing cross validation only with training data with the aim of getting the best tuning parameters
# no need to explicitly do the cross validation because it is done within the gbm function
for (shrink in shrinkages) {
  for (depth in depths) {
    set.seed(123)
    gbm.diabetes2 <- gbm(Diabetes_binary ~.,data=diabetestrain,
                     distribution = 'bernoulli', # bernoulli for binary classification
                     n.trees = 500, 
                     shrinkage = shrink, # this low shrinkage will tell me i need more trees?
                     interaction.depth = depth,
                     cv.folds = 5)
    
    perf_gbm2 = gbm.perf(gbm.diabetes2, method="cv", plot.it=FALSE) 
    
    y2hat2 <- ifelse(predict(gbm.diabetes2, newdata = diabetestrain[,-1], n.trees=perf_gbm2, type="response") < 0.5, 0, 1)
    gbm2_te <- mean(y2hat2 != diabetestrain$Diabetes_binary)
    cverror2 <- c(cverror2, gbm2_te);
    
    if (gbm2_te < best_gbm_te) {
      best_gbm_te <- gbm2_te
      best_params2 <- c(paste("shrinkage =",as.character(shrink),", interaction.depth =",as.character(depth)))
    }
  }
}

best_gbm_te # 0.115991
best_params2 # best params are "shrinkage = 0.05 , interaction.depth = 5" for 0.5%
# best params for 5% are also "shrinkage = 0.15 , interaction.depth = 5" for 20% of balanced

# make new model with best params
set.seed(123)
gbm.diabetes3 <- gbm(Diabetes_binary~.,data=diabetestrain,
                     distribution = 'bernoulli',
                     n.trees = 500, 
                     shrinkage = 0.15, 
                     interaction.depth = 5,
                     cv.folds = 5)
## Model Inspection 
## Find the estimated optimal number of iterations of trees - important so we don't overfit our training data 
# and we'll have poor performance on testing data
perf_gbm3 = gbm.perf(gbm.diabetes3, method="cv") 
perf_gbm3 # 70 at 0.5% # 42 at 20% of balanced

## summary model
## Which variances are important
summary(gbm.diabetes3)

## Make Prediction
## use "predict" to find the training or testing error

## Training error
pred3gbm <- predict(gbm.diabetes3, newdata = diabetestrain[,-1], n.trees=perf_gbm3, type="response")
pred3gbm[1:10]

y1hat3 <- ifelse(pred3gbm < 0.5, 0, 1)
y1hat3[1:10]
sum(y1hat3 != y1)/length(y1)  ##Training error = 0.115991

## Testing Error
y2hat3 <- ifelse(predict(gbm.diabetes3, newdata = diabetestest[,-1], n.trees=perf_gbm3, type="response") < 0.5, 0, 1)
mean(y2hat3 != y2) 
## Testing error = 0.1181102
######################################################################################################






######################################################################################################
# XGBOOST
# right now, I have as default values
set.seed(123)
xgboost1 <- xgboost(data=as.matrix(diabetestrain[,-1]), label = as.matrix(diabetestrain$Diabetes_binary), nrounds = 10, cv.folds = 5, objective="binary:logistic")

## Training error
predxgboost1 <- predict(xgboost1, newdata = as.matrix(diabetestrain[,-1]), type="response")
predxgboost1[1:10]

y1hat4 <- ifelse(predxgboost1 < 0.5, 0, 1)
y1hat4[1:10]
sum(y1hat4 != y1)/length(y1)  ##Training error = 0.06981982

## Testing Error
y2hat4 <- ifelse(predict(xgboost1, newdata = as.matrix(diabetestest[,-1]), type="response") < 0.5, 0, 1)
mean(y2hat4 != y2) 
## Testing error = 0.1312336


####now loop through multiple values of the parameters to see which is best
# build the boosting model from the training data
etas <- c(0.01, 0.05, 0.1, 0.2) # don't want too big eta values bc can lead to unstable behavior of model
depths <- c(3,4,5,6,7) # not too deep bc don't want to overfit
subsamples <- c(0.5, 0.6, 0.7, 0.8, 0.9)
cverror3 = NULL;
best_xg_te <- 1.0

# doing cross validation only with training data with the aim of getting the best tuning parameters
# no need to explicitly do the cross validation because it is done within the xgboost function
for (eta in etas){
  for (depth in depths) {
    for (sub in subsamples) {
      set.seed(123)
      xgboost2 <- xgboost(data=as.matrix(diabetestrain[,-1]), 
                          label = as.matrix(diabetestrain$Diabetes_binary), 
                          nrounds = 10, 
                          objective="binary:logistic",
                          eta = eta,
                          max_depth = depth,
                          subsample = sub)
      
      y2hat5 <- ifelse(predict(xgboost2, newdata = as.matrix(diabetestrain[,-1]), type="response") < 0.5, 0, 1)
      xgboost2_te <- mean(y2hat5 != diabetestrain$Diabetes_binary)
      cverror2 <- c(cverror2, xgboost2_te);
      
      if (xgboost2_te < best_xg_te) {
        best_xg_te <- xgboost2_te
        best_params3 <- c(paste("eta =",as.character(eta),", max_depth =",as.character(depth),", subsample =",as.character(sub)))
      }
    }
  }
}

best_xg_te # 0.05968468
best_params3 # "eta = 0.2 , max_depth = 7 , subsample = 0.9" for 0.5%
# best params are "eta = 0.2 , max_depth = 7 , subsample = 0.9" for 20% of balanced.

# make new model with best params
set.seed(123)
xgboost2 <- xgboost(data=as.matrix(diabetestrain[,-1]), label = as.matrix(diabetestrain$Diabetes_binary), nrounds = 10, cv.folds = 5, objective="binary:logistic", eta = 0.2, max_depth = 7, subsample = 0.9)

## summary model
## Which variances are important
summary(xgboost2)

# importance
xgboost_importance <- xgb.importance(model=xgboost2)
xgboost_importance

## Make Prediction
## use "predict" to find the training or testing error

## Training error
pred3xgboost <- predict(xgboost2, newdata = as.matrix(diabetestrain[,-1]), type="response")
pred3xgboost[1:10]

y1hat4 <- ifelse(pred3xgboost < 0.5, 0, 1)
y1hat4[1:10]
sum(y1hat4 != y1)/length(y1)  ##Training error = 0.05968468

## Testing Error
y2hat3 <- ifelse(predict(gbm.diabetes3, newdata = diabetestest[,-1], n.trees=perf_gbm3, type="response") < 0.5, 0, 1)
mean(y2hat3 != y2) 
## Testing error = 0.1181102
######################################################################################################







######################################################################################################
B= 50; ### number of loops
TEALL = NULL; ### Final Test Error values
TrEALL = NULL ### Final Training Error values
PrecisionALL = NULL;
RecallALL = NULL;
F1ALL = NULL;
AUCALL = NULL;
set.seed(123)
n = dim(sample_diabetes)[1]; ### total number of observations
n1 = round(n * 3 / 10); ### number of observations randomly selected for testing data

for (b in 1:B){
  set.seed(b)
  flag <- sort(sample(1:n, n1));
  train <- sample_diabetes[-flag,];
  test <- sample_diabetes[flag,];
  
  cverror <- NULL;
  cverror_train <- NULL;
  cvprecision <- NULL;
  cvrecall <- NULL;
  cvF1 <- NULL
  cvAUC <- NULL
  
  ytrue <- test$Diabetes_binary; 
  ytrue_tr <- train$Diabetes_binary; 
  ## A comparison with other methods
  ## Testing errors of several algorithms on the diabetes dataset:
  # using features from AIC feature selection: HighBP + HighChol + BMI + HeartDiseaseorAttack + GenHlth + MentHlth + Age
  # at 20% of balanced data --- using features from AIC feature selection: HighBP + HighChol + CholCheck + BMI + Stroke + HeartDiseaseorAttack + Veggies + HvyAlcoholConsump + AnyHealthcare + NoDocbcCost + GenHlth + PhysHlth + Sex + Age + Income
  set.seed(b)
  modA <- glm(as.factor(Diabetes_binary) ~ HighBP + HighChol + CholCheck + BMI + Stroke + HeartDiseaseorAttack + Veggies + HvyAlcoholConsump + AnyHealthcare + NoDocbcCost + GenHlth + PhysHlth + Sex + Age + Income, family = binomial(link="logit"), data=train)
  y2hatA <- ifelse(predict(modA, test[,-1], type="response" ) < 0.5, 0, 1)
  lr_te <- sum(y2hatA != ytrue)/length(ytrue)
  y2hatA_tr <- ifelse(predict(modA, train[,-1], type="response" ) < 0.5, 0, 1)
  lr_tre <- sum(y2hatA_tr != ytrue_tr)/length(ytrue_tr)
  lr_cm <- confusionMatrix(factor(y2hatA, levels=unique(c(y2hatA,ytrue))), factor(ytrue, levels=unique(c(y2hatA,ytrue))))
  cvprecision <- c(cvprecision, lr_cm$byClass["Pos Pred Value"]); 
  cvrecall <- c(cvrecall, lr_cm$byClass["Sensitivity"]); 
  cvF1 <- c(cvF1, 2*(lr_cm$byClass["Pos Pred Value"]*lr_cm$byClass["Sensitivity"])/(lr_cm$byClass["Pos Pred Value"]+lr_cm$byClass["Sensitivity"])); 
  cvAUC <- c(cvAUC, auc(roc(predictor=as.numeric(y2hatA), response=as.numeric(ytrue)))); 
  cverror <- c(cverror, lr_te); 
  cverror_train <- c(cverror_train, lr_tre); 
  summary(modA)
  
  #B.Linear Discriminant Analysis
  library(MASS)
  set.seed(b)
  modB <- lda(as.factor(Diabetes_binary) ~ HighBP + HighChol + CholCheck + BMI + Stroke + HeartDiseaseorAttack + Veggies + HvyAlcoholConsump + AnyHealthcare + NoDocbcCost + GenHlth + PhysHlth + Sex + Age + Income, data=train)
  y2hatB <- predict(modB, test[,-1])$class
  lda_te <- mean( y2hatB  != ytrue)
  y2hatB_tr <- predict(modB, train[,-1])$class
  lda_tre <- mean( y2hatB_tr  != ytrue_tr)
  cverror <- c(cverror, lda_te); 
  cverror_train <- c(cverror_train, lda_tre); 
  lda_cm <- confusionMatrix(factor(y2hatB, levels=unique(c(y2hatB,ytrue))), factor(ytrue, levels=unique(c(y2hatB,ytrue))))
  cvprecision <- c(cvprecision, lda_cm$byClass["Pos Pred Value"]); 
  cvrecall <- c(cvrecall, lda_cm$byClass["Sensitivity"]); 
  cvF1 <- c(cvF1, 2*(lda_cm$byClass["Pos Pred Value"]*lda_cm$byClass["Sensitivity"])/(lda_cm$byClass["Pos Pred Value"]+lda_cm$byClass["Sensitivity"])); 
  cvAUC <- c(cvAUC, auc(roc(predictor=as.numeric(y2hatB), response=as.numeric(ytrue)))); 
  
  
  ## C. Naive Bayes
  library(e1071)
  set.seed(b)
  modC <- naiveBayes(as.factor(Diabetes_binary) ~ HighBP + HighChol + CholCheck + BMI + Stroke + HeartDiseaseorAttack + Veggies + HvyAlcoholConsump + AnyHealthcare + NoDocbcCost + GenHlth + PhysHlth + Sex + Age + Income, data = train)
  y2hatC <- predict(modC, newdata = test[,-1])
  y2hatC_tr <- predict(modC, newdata = train[,-1])
  nb_te <- mean( y2hatC != ytrue) 
  nb_tre <- mean( y2hatC_tr  != ytrue_tr)
  cverror <- c(cverror, nb_te);
  cverror_train <- c(cverror_train, nb_tre); 
  nb_cm <- confusionMatrix(factor(y2hatC, levels=unique(c(y2hatC,ytrue))), factor(ytrue, levels=unique(c(y2hatC,ytrue))))
  cvprecision <- c(cvprecision, nb_cm$byClass["Pos Pred Value"]); 
  cvrecall <- c(cvrecall, nb_cm$byClass["Sensitivity"]); 
  cvF1 <- c(cvF1, 2*(nb_cm$byClass["Pos Pred Value"]*nb_cm$byClass["Sensitivity"])/(nb_cm$byClass["Pos Pred Value"]+nb_cm$byClass["Sensitivity"])); 
  cvAUC <- c(cvAUC, auc(roc(predictor=as.numeric(y2hatC), response=as.numeric(ytrue)))); 
  
  #D: a single Tree
  library(rpart)
  set.seed(b)
  modE0 <- rpart(as.factor(Diabetes_binary) ~ HighBP + HighChol + CholCheck + BMI + Stroke + HeartDiseaseorAttack + Veggies + HvyAlcoholConsump + AnyHealthcare + NoDocbcCost + GenHlth + PhysHlth + Sex + Age + Income,data=train, method="class", parms=list(split="gini"))
  opt <- which.min(modE0$cptable[, "xerror"]); 
  cp1 <- modE0$cptable[opt, "CP"];
  modE <- prune(modE0,cp=cp1);
  y2hatD <-  predict(modE, test[,-1],type="class")
  y2hatD_tr <-  predict(modE, train[,-1],type="class")
  st_te <- mean(y2hatD != ytrue)
  st_tre <- mean(y2hatD_tr != ytrue_tr)
  cverror <- c(cverror, st_te);
  cverror_train <- c(cverror_train, st_tre); 
  st_cm <- confusionMatrix(factor(y2hatD, levels=unique(c(y2hatD,ytrue))), factor(ytrue, levels=unique(c(y2hatD,ytrue))))
  cvprecision <- c(cvprecision, st_cm$byClass["Pos Pred Value"]); 
  cvrecall <- c(cvrecall, st_cm$byClass["Sensitivity"]); 
  cvF1 <- c(cvF1, 2*(st_cm$byClass["Pos Pred Value"]*st_cm$byClass["Sensitivity"])/(st_cm$byClass["Pos Pred Value"]+st_cm$byClass["Sensitivity"])); 
  cvAUC <- c(cvAUC, auc(roc(predictor=as.numeric(y2hatD), response=as.numeric(ytrue)))); 
  
  #E: random forest with my chosen parameters
  # best params are ntree=500, mtry=5, and nodesize=7, so use these to fit a new random forest model
  set.seed(b)
  rf3 <- randomForest(as.factor(Diabetes_binary) ~., data=train, ntree = 100 , mtry = 1 , nodesize = 4, importance=TRUE)
  y2hatE = predict(rf3, test[,-1], type='class')
  y2hatE_tr = predict(rf3, train[,-1], type='class')
  rf_te <- mean(y2hatE != ytrue)
  rf_tre <- mean(y2hatE_tr != ytrue_tr)
  cverror <- c(cverror, rf_te);
  cverror_train <- c(cverror_train, rf_tre); 
  rf_cm <- confusionMatrix(factor(y2hatE, levels=unique(c(y2hatE,ytrue))), factor(ytrue, levels=unique(c(y2hatE,ytrue))))
  cvprecision <- c(cvprecision, rf_cm$byClass["Pos Pred Value"]); 
  cvrecall <- c(cvrecall, rf_cm$byClass["Sensitivity"]); 
  cvF1 <- c(cvF1, 2*(rf_cm$byClass["Pos Pred Value"]*rf_cm$byClass["Sensitivity"])/(rf_cm$byClass["Pos Pred Value"]+rf_cm$byClass["Sensitivity"])); 
  cvAUC <- c(cvAUC, auc(roc(predictor=as.numeric(y2hatE), response=as.numeric(ytrue)))); 
  
  
  #F: gbm with my chosen parameters
  set.seed(b)
  gbm.diabetes3 <- gbm(Diabetes_binary~.,data=train,
                       distribution = 'bernoulli',
                       n.trees = 500, 
                       shrinkage = 0.15, 
                       interaction.depth = 5,
                       cv.folds = 5)
  ## Model Inspection 
  set.seed(b)
  perf_gbm3 = gbm.perf(gbm.diabetes3, method="cv",plot.it = FALSE) 
  y2hatF <- ifelse(predict(gbm.diabetes3, newdata = test[,-1], n.trees=perf_gbm3, type="response") < 0.5, 0, 1)
  y2hatF_tr <- ifelse(predict(gbm.diabetes3, newdata = train[,-1], n.trees=perf_gbm3, type="response") < 0.5, 0, 1)
  gbm_te <- mean(y2hatF != ytrue) 
  gbm_tre <- mean(y2hatF_tr != ytrue_tr) 
  cverror <- c(cverror, gbm_te);
  cverror_train <- c(cverror_train, gbm_tre); 
  gbm_cm <- confusionMatrix(factor(y2hatF, levels=unique(c(y2hatF,ytrue))), factor(ytrue, levels=unique(c(y2hatF,ytrue))))
  cvprecision <- c(cvprecision, gbm_cm$byClass["Pos Pred Value"]); 
  cvrecall <- c(cvrecall, gbm_cm$byClass["Sensitivity"]); 
  cvF1 <- c(cvF1, 2*(gbm_cm$byClass["Pos Pred Value"]*gbm_cm$byClass["Sensitivity"])/(gbm_cm$byClass["Pos Pred Value"]+gbm_cm$byClass["Sensitivity"])); 
  cvAUC <- c(cvAUC, auc(roc(predictor=as.numeric(y2hatF), response=as.numeric(ytrue)))); 
  
  #G: xbgboost with my chosen parameters
  set.seed(b)
  xgboost3 <- xgboost(data=as.matrix(diabetestrain[,-1]), 
                      label = as.matrix(diabetestrain$Diabetes_binary), 
                      nrounds = 10, 
                      objective="binary:logistic",
                      eta = 0.2,
                      max_depth = 7,
                      subsample = 0.9)
  
  y2hatG <- ifelse(predict(xgboost3, newdata = as.matrix(test[,-1]), type="response") < 0.5, 0, 1)
  y2hatG_tr <- ifelse(predict(xgboost3, newdata = as.matrix(train[,-1]), type="response") < 0.5, 0, 1)
  xgb_te <- mean(y2hatG != ytrue)
  xgb_tre <- mean(y2hatG_tr != ytrue_tr)
  cverror <- c(cverror, xgb_te);
  cverror_train <- c(cverror_train, xgb_tre);
  xg_cm <- confusionMatrix(factor(y2hatG, levels=unique(c(y2hatG,ytrue))), factor(ytrue, levels=unique(c(y2hatG,ytrue))))
  cvprecision <- c(cvprecision, xg_cm$byClass["Pos Pred Value"]); 
  cvrecall <- c(cvrecall, xg_cm$byClass["Sensitivity"]); 
  cvF1 <- c(cvF1, 2*(xg_cm$byClass["Pos Pred Value"]*xg_cm$byClass["Sensitivity"])/(xg_cm$byClass["Pos Pred Value"]+xg_cm$byClass["Sensitivity"])); 
  cvAUC <- c(cvAUC, auc(roc(predictor=as.numeric(y2hatG), response=as.numeric(ytrue)))); 

  TEALL = rbind(TEALL, cverror);
  TrEALL = rbind(TrEALL, cverror_train);
  PrecisionALL = rbind(PrecisionALL, cvprecision);
  RecallALL = rbind(RecallALL, cvrecall);
  F1ALL = rbind(F1ALL, cvF1);
  AUCALL = rbind(AUCALL, cvAUC);
}

dim(TEALL); ### This should be a Bx6
colnames(TEALL) <- c("log-reg", "lda", "naive-bayes", "singletree", "randomforest", "gbm", "xgboost");
colnames(TrEALL) <- c("log-reg", "lda", "naive-bayes", "singletree", "randomforest", "gbm", "xgboost");
colnames(PrecisionALL) <- c("log-reg", "lda", "naive-bayes", "singletree", "randomforest", "gbm", "xgboost");
colnames(RecallALL) <- c("log-reg", "lda", "naive-bayes", "singletree", "randomforest", "gbm", "xgboost");
colnames(F1ALL) <- c("log-reg", "lda", "naive-bayes", "singletree", "randomforest", "gbm", "xgboost");
colnames(AUCALL) <- c("log-reg", "lda", "naive-bayes", "singletree", "randomforest", "gbm", "xgboost");

## You can report the sample mean and sample variances for the 6 models
mean_tre <- apply(TrEALL, 2, function(x) mean(x, na.rm=TRUE));
var_tre <- apply(TrEALL, 2, function(x) var(x, na.rm=TRUE));

mean_te <- apply(TEALL, 2, function(x) mean(x, na.rm=TRUE));
var_te <- apply(TEALL, 2, function(x) var(x, na.rm=TRUE));

mean_precision <- apply(PrecisionALL, 2, function(x) mean(x, na.rm=TRUE));
var_precision <- apply(PrecisionALL, 2, function(x) var(x, na.rm=TRUE));

mean_recall <- apply(RecallALL, 2, function(x) mean(x, na.rm=TRUE));
var_recall <- apply(RecallALL, 2, function(x) var(x, na.rm=TRUE));

mean_f1 <- apply(F1ALL, 2, function(x) mean(x, na.rm=TRUE));
var_f1 <- apply(F1ALL, 2, function(x) var(x, na.rm=TRUE));

mean_auc <- apply(AUCALL, 2, function(x) mean(x, na.rm=TRUE));
var_auc <- apply(AUCALL, 2, function(x) var(x, na.rm=TRUE));

best_tre <- sort((mean_tre))
best_tre

best_te <- sort((mean_te))
best_te

best_precision <- sort((mean_precision))
best_precision

best_recall <- sort((mean_recall))
best_recall

best_f1 <- sort((mean_f1))
best_f1

best_auc <- sort((mean_auc))
best_auc

best_var_te <- sort((var_te))
best_var_te

best_var_precision <- sort((var_precision))
best_var_precision

best_var_recall <- sort((var_recall))
best_var_recall

best_var_f1 <- sort((var_f1))
best_var_f1

best_var_auc <- sort((var_auc))
best_var_auc
######################################################################################################





