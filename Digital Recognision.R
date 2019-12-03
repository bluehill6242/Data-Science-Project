####################### Hand Written Digit Reconisation #############################
#----------------------------- Support Vector Machine ------------------------------#
#
# CRISP-DM approach followed to address the Pattrn Recognisation for Hand Written
# Digit Recognisation on MNIST dataset. 
#
############################ Business understanding #################################
#
# MNIST Database stands for the Modified National Institute of Standards and
# Technology. It is a curated database of handwritten digits used for computer vision,
# pattern recognition and to train Machine Learning Algorithms. The database contains
# images of handwritten digits each with varying writing styles.
#
# -------------------------- Business Objective ----------------------------------- #
#
# We have an image of a digit submitted by a user via a scanner, a tablet, or other
# digital devices. The goal is to develop a model that can correctly identify the
# digit (between 0-9) written in an image Using Support Vector Machine which should
# correctly classify the handwritten digits based on the pixel values given as features.
#
# ------------------------- Goal of Data Analysis ----------------------------------#
#
# The goal of the data analysis is to develop a SVM model applying concept of 
# hyperlane and kernels that can identify the digits (between 0-9) written in different
# writting styles and submitted as image via scanner, tablet or nay other digital device.
#
############################### Data Understanding ##################################
#
# Source of data - MNIST Database
# Data Description and preperation needs
# Training Dataset  - has 60,000 images of handwritten digits
# Testing Dataset - has 10,000 images of handwritten digits.
# 
# Each image is normalized to a 28rows x 28column pixel matrix.
# The dataset generated from these images has 784 attributes with each attribute
# representing one element of the pixel matrix. Each record of the 784 attributes is
# filled in with a value between 0-255 both inclusive, these values are representative
# of the ink density of the corresponding pixel from the image. Pixels are organized
# row-wise. Pixel values are 0 to 255. 0 means background (white), 255 means
# foreground (black). The Objective of this assignent is to implement SVM algorithm
# to create a multiclassifier to distinguish these numbers based on the pixel
# information. We will also conduct hyperparameter tuning using 3-fold cross
# validation. For a detailed understanding of the dataset please refer to the 
# following website: http://yann.lecun.com/exdb/mnist/
#
################################## Data Preperation #################################
# 
# Load Required Libraries
# install pacman package to load and/or install libraires
# reference - https://cran.r-project.org/web/packages/pacman/vignettes/Introduction_to_pacman.html

install.packages("pacman")

pacman::p_load(dplyr, ggplot2, stringr, tidyr, kernlab, readr, e1071, stats, 
               gridExtra, caret, caTools)

# Set working directory to read data

# setwd("D:/Lectures/Data Science/Upgrad/Course III - Predictive Analytics/SVM Assignment")

# loading data to R and check structure for dataset

test_data <- read.csv("mnist_test.csv",sep = ",", stringsAsFactors = F,header = F)
glimpse(test_data)      # observation - 9,999, varibles - 785(int type)
dim(test_data)

train_data <- read.csv("mnist_train.csv",sep = ",", stringsAsFactors = F,header = F)
glimpse(train_data)     # observation - 59,999, varibles - 785(int type)
dim(train_data)

# Training database has 60000 records with each record having 785 attributes.
# Each of the 60000 records corresponds to the pixel information of 1 handwritten digit. 
# Since each digit is captured in a 28 rows X 28 columns image there are 28 X 28 = 784 attributes
# that correspond to pixel information and 1 attribute containg the label of the Digit.

# Both test_data and train_data datasets do not have column names.
# We will rename the first column of both datasets as "Number"
# "Number" is the target variable.

colnames(train_data)[1]<- "Number"
colnames(test_data)[1]<- "Number"

# Check for data type other than integer in train_data dataset. Output shows 0 columns with
# other data type than integer

train_data[, lapply(train_data[,-1], is.numeric) == FALSE] 

# Check for data type other than integer in test_data dataset. Output shows 0 columns with
# other data type than integer

test_data[, lapply(test_data[,-1], is.numeric) == FALSE] 

# Print a number to see how the data is organized
number <- matrix(as.numeric(train_data[1,-1]), nrow = 28) #look at one digit
image(number, col = grey.colors(255))

# Check for NA in train and test Dataset. 0 represent no NA values.

sum(is.na(train_data))
sum(is.na(test_data))

# Check the frequency of each label in train data

train_data$intensity <- apply(train_data[,-1], 1, mean) #takes the mean of each row in train

label_intensity <- aggregate (train_data$intensity, by = list(train_data$Number), FUN = mean)

plot <- ggplot(data=label_intensity, aes(x=Group.1, y = x)) +
  geom_bar(stat="identity")
plot + scale_x_discrete(limits=0:9) + xlab("digit label") + 
  ylab("average intensity")

rm(plot)
rm(label_intensity)

# Check the frequency of each label in train data

test_data$intensity <- apply(test_data[,-1], 1, mean) #takes the mean of each row in train

label_intensity <- aggregate (test_data$intensity, by = list(test_data$Number), FUN = mean)

plot <- ggplot(data=label_intensity, aes(x=Group.1, y = x)) +
  geom_bar(stat="identity")
plot + scale_x_discrete(limits=0:9) + xlab("digit label") + 
  ylab("average intensity")

rm(plot)
rm(label_intensity)

# remove intesity from test and train dataset

train_data[,786] <- NULL
test_data[,786] <- NULL

# Data Cleansing & Transformations & Feature Engineering

# Check for columns having same value for all records in train dataset

length(unique(train_data[sapply(train_data, function(x) length(unique(x)) == 1)]))

# removing columns having 0s for each of records using nearZeroVAr
# Keeping threshold at 100% for this and identify those columns.
# for train data

cols_name_1 <- nearZeroVar(train_data, freqCut = 100/0, names = TRUE , uniqueCut = 0)
length(cols_name_1)     # length is same as above checked using unique function
train_data <- train_data[, -which(names(train_data) %in% c(cols_name_1))]

# Check for columns having same value for all records in train dataset

length(unique(test_data[sapply(test_data, function(x) length(unique(x)) == 1)]))

# Since test data have 116 columns with all zeros. removing only 67 columns from test
# dataset as in train dataseet

test_data <- test_data[, -which(names(test_data) %in% c(cols_name_1))]

# Changing output variable "Number" to factor type 

train_data$Number <- factor(train_data$Number)

test_data$Number <- factor(test_data$Number)


# Take 10K sample from training set for ksvm training set & 2K sample for Cross Validation
# Sample from Test set is taken as the cross validation requires high cumputational resourse

set.seed(100)
train_sample_1 <- sample(1:nrow(train_data), 10000)

set.seed(100)
train_sample_2 <- sample(1:nrow(train_data), 2000)


train_data_1 <- train_data[train_sample_1,]    # 10000 observation, 718 variables 
train_data_2 <- train_data[train_sample_2,]   # 2000 observation, 718 variables

############################### Model Building ###############################################

# Linear SVM 

# KSVM model building. Build with C=1

model_1 <- ksvm(Number ~ ., data = train_data_1, scale = FALSE, C = 1)
model_1

# Default Parameters Penalty for Misclassification Cost [C=1]
# Number of support vectors influencing decision boundary = 3800


# Predicting the model results 

model_1_predict <- predict(model_1, newdata = test_data)

# Confusion Matrix - Finding accuracy, Sensitivity and specificity

model_1_conf_Mat <- confusionMatrix(model_1_predict, test_data$Number)
model_1_conf_Mat

# KSVM model building with C = 10

model_2 <- ksvm(Number ~ ., data = train_data_1, scale = FALSE, C = 10)

# Predicting the model results 

model_2_predict <- predict(model_2, newdata = test_data)

# Confusion Matrix - Finding accuracy, Sensitivity and specificity

model_2_conf_Mat <- confusionMatrix(model_2_predict, test_data$Number)
model_2_conf_Mat

model_1_conf_Mat$overall
model_2_conf_Mat$overall


# The above two models shows that Linear SVM mode at C=10 is slightly better compared to C=1.

# Modeling- Non-Linear SVM using Kernels.
# Non Linear SVM models - RBF Kernel

model_3 <- ksvm(Number ~ ., data = train_data_1, scale = FALSE, kernel = "rbfdot")

model_3_predict <- predict(model_3, newdata = test_data)

# confusion matrix - RBF Kernel

model_3_conf_Mat <- confusionMatrix(model_3_predict,test_data$Number)
print(model_3_conf_Mat)

## RBF Kernel 


# Making grid of "sigma" and C values. 

tc <- trainControl(method="cv", number=5)
metric <- "Accuracy"
set.seed(80)
grid <- expand.grid(.sigma=seq(0.01, 0.05, by=0.01), .C=seq(1, 5, by=1))
grid


# Performing 5-fold cross validation for on 2000 variables 

fit.svmRadial <- train(Number ~ ., data = train_data_2, method = "svmRadial", metric = metric, 
                        tuneGrid = grid, trControl = tc)

# Printing cross validation result

print(fit.svmRadial)

# Best tune at sigma = 0.05 & C=1, Accuracy - 0.935

# Plotting model results
plot(fit.svmRadial)



## Conclusions:
## In my case, due to the way samples have been take, linear SVM at C=2 produced best results compared to all others