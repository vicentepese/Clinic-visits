library(jsonlite)
library(dplyr)
library(corrplot)
library(PerformanceAnalytics)
library(lme4)
library(ggplot2)
library(FeatureHashing)
library(nlme)

########## INITIALIZATION ##########

# Set working directory
setwd("~/Documents/Vitalbeats/ShowNoShow")

# Load settings 
settings <- jsonlite::read_json('settings.json')

# Load data 
data.df <- read.csv(settings$file$data)

# Convert variables to factor and dates
data.df$Neighbourhood <- data.df$Neighbourhood %>% as.factor()
data.df$No.show <- data.df$No.show %>% as.factor() %>% as.numeric() -1 
data.df$Gender <- data.df$Gender %>% as.factor()
data.df$AppointmentID <- data.df$AppointmentDay %>% as.factor()
data.df$PatientId <- data.df$PatientId %>% as.factor()
data.df$hourSchedule <- data.df$ScheduledDay %>% lapply(function(x) x %>% strsplit(split = 'T') %>% unlist() %>% 
                                                          .[2] %>% strsplit(split = 'Z') %>% unlist() %>% .[1]) %>% unlist()
data.df$hourSchedule <- format(data.df$hourSchedule %>% strptime("%H:%M:%S"), "%H") %>% as.factor()
data.df$ScheduledDay <- data.df$ScheduledDay %>% as.Date()


############# DATA EXPLORATION #############

# Cound number of show/no show instances
SNS.count <- data.df$No.show %>% table() %>% prop.table() *100
SNS.count

# GGplot of No.show between variables
ggplot(data.df, aes(y = No.show, x = Gender, color = Gender)) + geom_col()

# Correlation plot of numeric data 
numData.df <- select_if(data.df, is.numeric)
chart.Correlation(numData.df, histogram=TRUE, pch = 19)
corrplot(cor(numData.df))

############ Mixed model glm #############

# Convert categorical variables through hashing trick 
data.df$PatientId <- FeatureHashing::hashed.value(data.df$PatientId)
data.df$hourSchedule <- hashed.value(data.df$hourSchedule)
data.df$ScheduledDay <- hashed.value(data.df$ScheduledDay)
data.df$AppointmentDay <- hashed.value(data.df$AppointmentDay)
data.df$Neighbourhood <- hashed.value(data.df$Neighbourhood)

# Split in train_test:
balance <- FALSE
if (balance == TRUE){
  
  # Sample 
  data.df <- data.df[sample(1:nrow(data.df)),]
  
  # Separate by labels
  data.NS1 <- data.df %>% filter(No.show == 1); data.NS1 <- data.NS1[sample(1:nrow(data.NS1)),]
  data.NS0 <- data.df %>% filter(No.show == 0); data.NS0 <- data.NS0[sample(1:nrow(data.NS0)),]
  
  # Compute min rows and take 
  min.obs <- min(c(nrow(data.NS1), nrow(data.NS0)))
  
  # Take values 
  data.NS1 <- data.NS1[1:min.obs,]
  data.NS0 <- data.NS0[1:min.obs,]
  
  # Merge and sample 
  data.merge <- rbind(data.NS1, data.NS0) %>% .[sample(1:(2*nrow(data.NS0))),]
  
  # Split 
  tr_len <- round(nrow(data.merge)*0.8)
  data.train <- data.merge[c(1:tr_len),]
  data.test <- data.merge[c(tr_len:nrow(data.merge)),]
  
} else{
  
  # Sample 
  data.df <- data.df[sample(1:nrow(data.df)),]
  
  # Split 
  tr_len <- round(nrow(data.df)*0.8)
  data.train <- data.df[c(1:tr_len),]
  data.test <- data.df[c(tr_len:nrow(data.df)),]
  
}

# Fit mixed model
model <- lme(No.show ~ Gender + ScheduledDay + AppointmentDay + Age + Neighbourhood + Scholarship +
       Hipertension + Diabetes + Alcoholism + Handcap + SMS_received + hourSchedule , random=~1|PatientId, data = data.train)

# Predict 
y_pred <- predict(model, data.train)
y_test <- data.train$No.show
y_pred <- round(y_pred)
y_pred <- y_pred %>% unname()
acc <- y_pred[!is.na(y_pred)] == y_test[!is.na(y_pred)] 
acc %>% table() %>% prop.table() *100

