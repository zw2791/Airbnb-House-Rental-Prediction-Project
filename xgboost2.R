#xgboosting
setwd('C:/Users/sayua/OneDrive/Desktop/kaggle data')
library(caret)
library(lattice)
library(survival)
library(Formula)
library(ggplot2)
library(Hmisc)

# read data
train_data = read.csv('analysisData.csv',encoding="UTF-8")
scoring_data = read.csv('scoringData.csv',encoding="UTF-8")
str(train_data); head(train_data)

# replace Na value
# weekly_price
train_data[is.na(train_data$weekly_price),"weekly_price"] = 
  mean(train_data[!is.na(train_data$weekly_price),"weekly_price"])
# beds
train_data[is.na(train_data$beds),"beds"] = 
  mean(train_data[!is.na(train_data$beds),"beds"])
# square_feet
train_data[is.na(train_data$square_feet),"square_feet"] = 
  mean(train_data[!is.na(train_data$square_feet),"square_feet"])
# security_deposit
train_data[is.na(train_data$security_deposit),"security_deposit"] = 
  mean(train_data[!is.na(train_data$security_deposit),"security_deposit"])
# cleaning_fee
train_data[is.na(train_data$cleaning_fee),"cleaning_fee"] = 
  mean(train_data[!is.na(train_data$cleaning_fee),"cleaning_fee"])
# reviews_per_month
train_data[is.na(train_data$reviews_per_month),"reviews_per_month"] = 
  mean(train_data[!is.na(train_data$reviews_per_month),"reviews_per_month"])
# host_total_listings_count
train_data[is.na(train_data$host_total_listings_count),"host_total_listings_count"]= mean(train_data[!is.na(train_data$host_total_listings_count),"host_total_listings_count"])
# host_response_time
train_data[train_data$host_response_time=="","host_response_time"] = "N/A"


# select variables
library(dplyr)
train = train_data%>%
  group_by()%>%
  select(price,neighbourhood_group_cleansed,is_location_exact,accommodates,bathrooms,bedrooms,beds,
           security_deposit,
           cleaning_fee, zipcode,
           guests_included,
           extra_people,
           minimum_nights, maximum_nights,
           number_of_reviews,
           number_of_reviews_ltm,
           review_scores_rating,
           review_scores_accuracy,review_scores_cleanliness,
           review_scores_checkin,review_scores_communication,
           review_scores_location,review_scores_value,
           cancellation_policy,
           reviews_per_month,
           weekly_price,
           calculated_host_listings_count_shared_rooms, calculated_host_listings_count_entire_homes,
           calculated_host_listings_count_private_rooms,
           availability_30,availability_60,
           availability_90,availability_365,
           calendar_updated,
           host_listings_count, host_acceptance_rate, host_has_profile_pic,
           neighbourhood,host_neighbourhood,
           host_total_listings_count,host_response_time,neighbourhood_cleansed)

test = scoring_data%>%
  group_by()%>%
  select(neighbourhood_group_cleansed,is_location_exact,accommodates,bathrooms,bedrooms,beds,
         security_deposit,
         cleaning_fee, zipcode,
         guests_included,
         extra_people,
         minimum_nights, maximum_nights,
         number_of_reviews,
         number_of_reviews_ltm,
         review_scores_rating,
         review_scores_accuracy,review_scores_cleanliness,
         review_scores_checkin,review_scores_communication,
         review_scores_location,review_scores_value,
         cancellation_policy,
         reviews_per_month,
         weekly_price,
         calculated_host_listings_count_shared_rooms, calculated_host_listings_count_entire_homes,
         calculated_host_listings_count_private_rooms,
         availability_30,availability_60,
         availability_90,availability_365,
         calendar_updated,
         host_listings_count, host_acceptance_rate, host_has_profile_pic,
         neighbourhood,host_neighbourhood,
         host_total_listings_count,host_response_time,neighbourhood_cleansed)

# data preparation through vtreat
library(vtreat)
set.seed(9529)
trt = designTreatmentsZ(dframe = train,
                        varlist = names(train)[2:42])
newvars = trt$scoreFrame[trt$scoreFrame$code%in% c('clean','lev'),'varName']
train_input = prepare(treatmentplan = trt, 
                      dframe = train,
                      varRestriction = newvars)
test_input = prepare(treatmentplan = trt, 
                     dframe = test,
                     varRestriction = newvars)
head(train_input)

# cross validation
library(xgboost); library(caret)
tune_nrounds = xgb.cv(data=as.matrix(train_input), 
                      label = train$price,
                      nrounds=267,
                      nfold = 12,
                      verbose = 1)

# plot
ggplot(data=tune_nrounds$evaluation_log, aes(x=iter, y=test_rmse_mean))+
  geom_point(size=0.4, color='sienna')+
  geom_line(size=0.1, alpha=0.1)+
  theme_bw()
which.min(tune_nrounds$evaluation_log$test_rmse_mean)

# boosting
xgboost2= xgboost(data=as.matrix(train_input), 
                  label = train$price,
                  nrounds= 500,
                  verbose = 1,
                  objective = "reg:squarederror",
                  metrics = "rmse",
                  learning_rate = 0.05)
pred = predict(xgboost2, 
               newdata=as.matrix(test_input))

# calculate RMSE
rmse_xgboost = sqrt(mean((pred - train$price)^2)); rmse_xgboost

# submission
submission_data = data.frame(id = scoring_data$id, price = pred)
write.csv(submission_data, 'modelF1.csv',row.names = F)





























