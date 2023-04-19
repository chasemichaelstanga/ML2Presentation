setwd("C:/Users/jamie/SpringMSBA/MachineLearning2/Final Presentation")
rm(list=ls())

library(readr) # CSV file I/O, e.g. the read_csv function
library(tidyverse)
library(mlr)
library(xgboost)
library(lubridate)
library(knitr)
library(geosphere)

train <- read.csv("train.csv")
test <- read.csv("test.csv")


##Feature Engineering

#look at data
head(train,10)
head(test, 10)


train$pickup_month <- month(train$pickup_datetime)
train$pickup_day <- day(train$pickup_datetime)
train$pickup_hour <- hour(train$pickup_datetime)
train$pickup_min <- minute(train$pickup_datetime)

train$dropoff_month <- month(train$dropoff_datetime)
train$dropoff_day <- day(train$dropoff_datetime)
train$dropoff_hour <- hour(train$dropoff_datetime)
train$dropoff_min <- minute(train$dropoff_datetime)

dev.off()
train$trip_duration_min <- train$trip_duration/60
train$weekday <- wday(train$pickup_datetime)
train$weekend <- cut(train$weekday,breaks=c(0,5,7),labels= c("weekday","weekend"))

# No change to Latitude, only Longitude, result is in Meters
train$long_dist_man <- distHaversine(train[,c("pickup_longitude","pickup_latitude")], train[,c("dropoff_longitude","pickup_latitude")])

# No change to Longitude, only to Latitude, result in Meters again
train$lat_dist_man <- distHaversine(train[,c("pickup_longitude","pickup_latitude")], train[,c("pickup_longitude","dropoff_latitude")])

# Compute bootstrapped Manhattan Distance as absolute sum of Long/Lat distance
train$dist_man <- abs(train$lat_dist_man) + abs(train$long_dist_man)


#test
test$dropoff_datetime <- as.POSIXct(NA)
test$trip_duration <- 0

test$pickup_month <- month(test$pickup_datetime)
test$pickup_day <- day(test$pickup_datetime)
test$pickup_hour <- hour(test$pickup_datetime)
test$pickup_min <- minute(test$pickup_datetime)

test$dropoff_month <- month(test$dropoff_datetime)
test$dropoff_day <- day(test$dropoff_datetime)
test$dropoff_hour <- hour(test$dropoff_datetime)
test$dropoff_min <- minute(test$dropoff_datetime)

dev.off()
test$trip_duration_min <- test$trip_duration/60
test$weekday <- wday(test$pickup_datetime)
test$weekend <- cut(test$weekday,breaks=c(0,5,7),labels= c("weekday","weekend"))

# No change to Latitude, only Longitude, result is in Meters
test$long_dist_man <- 0
# No change to Longitude, only to Latitude, result in Meters again
test$lat_dist_man <- 0
# Compute bootstrapped Manhattan Distance as absolute sum of Long/Lat distance
test$dist_man <- 0

#write.csv(test, "test_new.csv", row.names = FALSE)
#write.csv(train, "train_new.csv", row.names = FALSE)

#### XGBOOST ####
traindata <- train %>% select(-pickup_datetime, -dropoff_datetime, -lat_dist_man, -long_dist_man,
                             -store_and_fwd_flag, -trip_duration_min, -dropoff_month, -dropoff_day,
                             -dropoff_hour, -dropoff_min)
testdata <- test %>% select(-pickup_datetime, -dropoff_datetime, -lat_dist_man, -long_dist_man,
                             -store_and_fwd_flag, -trip_duration_min, -dropoff_month, -dropoff_day,
                             -dropoff_hour, -dropoff_min)


#Convert data to DMatrix
traindata_x <- traindata %>% select(-id, -trip_duration)
##why take the log?
traindata_y <- log(traindata$trip_duration)

testdata_x <- testdata %>% select(-id, -trip_duration)

dtrain <- xgb.DMatrix(data=data.matrix(traindata_x), label=traindata_y)
dtest <- xgb.DMatrix(data=data.matrix(testdata_x))
watchlist <- list(traindata=dtrain)


#Setting Hyperparameters in XGBoost
nfolds <- 3
nrounds <- 50

params <- list("eta"=0.3,
               "max_depth"=10,
               "booster" = "gbtree",
               "colsample_bytree"=0.3,
               "min_child_weight"=1,
               "subsample"=0.8,
               "eval_metric"= "rmse", 
               "objective"= "reg:linear")


model_xgb <- xgb.train(params=params,
                       data=dtrain,
                       nrounds=nrounds,
                       maximize=FALSE,
                       watchlist=watchlist,
                       print_every_n=3)

predicted <- predict(model_xgb, dtest)
predicted_abs <- predicted %>% exp() %>% abs()


xgb_results <- cbind(testdata %>% select(id), trip_duration=predicted_abs)

model <- xgb.dump(model_xgb, with_stats=T)
model[1:10]
feature_names <- dimnames(data.matrix(traindata_x[,]))[[2]]
feature_names

importance <- xgb.importance(feature_names, model=model_xgb)
xgb.plot.importance((importance[1:10,]))
