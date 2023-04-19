rm(list = ls())

library(dplyr)

# Load the data into a data frame
data <- read.csv("train.csv")

# Sample 10,000 rows randomly
sampled_data <- data %>% sample_n(1000)

library(caret)

# Set the seed for reproducibility
set.seed(123)

# Split the data into training and test sets
train_index <- createDataPartition(sampled_data$trip_duration, p = 0.9, list = FALSE)
train_data <- sampled_data[train_index, ]
test_data <- sampled_data[-train_index, ]
#install.packages("randomForest")
library(randomForest)


# Separate month, day, and hour, and minute from the date/time column, into their own features

train$pickup_month <- month(train$pickup_datetime)
train$pickup_day <- day(train$pickup_datetime)
train$pickup_hour <- hour(train$pickup_datetime)
train$pickup_min <- minute(train$pickup_datetime)


# Same for drop off, creating 4 new features from the date-time column
train$dropoff_month <- month(train$dropoff_datetime)
train$dropoff_day <- day(train$dropoff_datetime)
train$dropoff_hour <- hour(train$dropoff_datetime)
train$dropoff_min <- minute(train$dropoff_datetime)


#converts trip duration into minutes
train$trip_duration_min <- train$trip_duration/60

#labels day of the week as numbers
train$weekday <- wday(train$pickup_datetime)

#creates a column that labels the day of the week - weekday or weekend
train$weekend <- cut(train$weekday,breaks=c(0,5,7),labels= c("weekday","weekend"))


#Calculate haversine distance of longitude and latitude

train$long_dist_man <- distHaversine(train[,c("pickup_longitude","pickup_latitude")], train[,c("dropoff_longitude","pickup_latitude")])

# No change to Longitude, only to Latitude, result in Meters again
train$lat_dist_man <- distHaversine(train[,c("pickup_longitude","pickup_latitude")], train[,c("pickup_longitude","dropoff_latitude")])

# Compute bootstrapped Manhattan Distance as absolute sum of Long/Lat distance
train$dist_man <- abs(train$lat_dist_man) + abs(train$long_dist_man)


#Doing the exact same thing, but for test, creates the column, but doesn't fill 
# it in because that's what we are creating in our model
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

#dev.off()
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



############## Build the random forest model ################
# used a for loop to tune this but it takes so long to run that we entered the best tuning features here
rf_model <- randomForest(trip_duration ~ ., data = train_data, ntree = 100, mtry=7)

# Make predictions on the test data
test_preds <- predict(rf_model, newdata = test_data)
test_preds
# Evaluate the model performance
rmse <- sqrt(mean((test_preds/60 - test_data$trip_duration/60)^2))
rmse # ON average, the duration predictions were off by 462 seconds, or roughly 7 1/2 minutes

# A RMSE (root mean squared error) of 1546 means that on average, the random forest 
# models predictions for trip duration are off by 1546 seconds (or approximately 25 
# minutes and 46 seconds).

# In other words, the models predicted trip duration values deviate from the 
# actual trip duration values by an average of 1546 seconds. A lower RMSE would 
# indicate better performance


