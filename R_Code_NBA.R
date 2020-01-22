#Prediction of NBA player 5-year career longevity with logistic regression

# Install the required packages and load the libraries
require(data.table)
require(dplyr)
require(SmartEDA)
require(InformationValue)
require(smbinning)

library(data.table)
library(dplyr)
library(SmartEDA)
library(smbinning)
library(InformationValue)

# --------------------- Step 1 - Load and understand the data ---------------------

# Import and Read the data
data_nba <- read.csv("nba_logreg.csv")

#Glimpse of the dataset
head(data_nba)

# Structure of the dataset
str(data_nba)

# Dimensions of the dataset
dim(data_nba)

# EDA for the dataset given
ExpData(data=data_nba, type=1)

# Handling the missing values
nba_missing = data_nba %>% 
  summarise_all((funs(sum(is.na(.))))) 

nba_missing   

# Handling missing values for the feature that has null values
data_nba$X3P. <- ifelse(is.na(data_nba$X3P.),
                              ave(data_nba$X3P.,FUN = function(x) mean(x,na.rm = TRUE)),
                        data_nba$X3P.)

# EDA of the data given after cleaning 
ExpData(data=data_nba, type=1)

# Check the target proportion 
table(data_nba$TARGET_5Yrs) # 0's and 1's count for the response variable (TARGET_5Yrs)

#Summary of the dataset
summary(data_nba)

#--------------------- Step 2 - Modeling of the Data ---------------------

# Create Training and Test Datasets

# Training Data
nba_ones <- data_nba[which(data_nba$TARGET_5Yrs == 1), ]  # all 1's - player having a career of at least five years
nba_zeros <- data_nba[which(data_nba$TARGET_5Yrs == 0), ]  # all 0's - players having career less than 5 yrs
set.seed(150)  # for repeatability of samples for making partition reproducible

nba_ones_training_rows <- sample(1:nrow(nba_ones), 0.8*nrow(nba_ones), replace = TRUE,)  # 1's for training dataset
nba_zeros_training_rows <- sample(1:nrow(nba_zeros), 0.8*nrow(nba_ones), replace = TRUE,)  # 0's for training dataset same as 1's

train_ones <- nba_ones[nba_ones_training_rows, ]  
train_zeros <- nba_zeros[nba_zeros_training_rows, ]
nba_train <- rbind(train_ones, train_zeros)  # training dataset with 1's and 0's 


# Test Data
test_ones <- nba_ones[-nba_ones_training_rows, ]
test_zeros <- nba_zeros[-nba_zeros_training_rows, ]
nba_test <- rbind(test_ones, test_zeros)  # test dataset with 1's and 0's of target variable

# Count of Train anad Test Dataset
nrow(nba_train)
nrow(nba_test)
nrow(data_nba)

# Model Development
nba_model <- glm(TARGET_5Yrs ~ GP + MIN + PTS + FGM + FGA + X3P.Made +	X3PA	+ FTM +	FTA +	OREB +	DREB +	REB +	AST	+STL +	BLK +	TOV, family=binomial(link=logit), data = nba_train)

summary(nba_model)

exp(coef(nba_model))

#--------------------- Step 3 - Prediction ---------------------

# Predicting the model with Test Data
predicted <- predict(nba_model, nba_test, type="response")  

target_pred <- ifelse(predicted > 0.5, 1, 0)
target_act <- nba_test$TARGET_5Yrs

misClasificError <- mean(target_pred != target_act) 
print(paste('Accuracy of the model is', 1-misClasificError)) # Accuracy 

# Calcuate the Optimal Cutoff
optCutOff <- optimalCutoff(nba_test$TARGET_5Yrs, predicted)[1] 

optCutOff # Print the cutoff

# Calculate the misClassError
misClassError(nba_test$TARGET_5Yrs, predicted, threshold = optCutOff)

# Plot the ROC Curve
plotROC(nba_test$TARGET_5Yrs, predicted)

# Calculating the concordance, sensitivity and specificity 
Concordance(nba_test$TARGET_5Yrs, predicted)

sensitivity(nba_test$TARGET_5Yrs, predicted, threshold = optCutOff)

specificity(nba_test$TARGET_5Yrs, predicted, threshold = optCutOff)

# Confusion Matrix
confusionMatrix(nba_test$TARGET_5Yrs, predicted, threshold = optCutOff)

