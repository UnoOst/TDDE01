set.seed(1234567890)
library(ggplot2)
library(geosphere)
library(dplyr)
library(knitr)
library(kableExtra)
library(readr)
#1 Implement Kernel Method to predict hourly temp/day (4am -24pm)


# Read data
station_data <- read.csv("stations.csv")
temp_data <- read.csv("temps50k.csv")

# MErge two files with respect to station_number

st <- merge(station_data, temp_data, by = "station_number")
n = dim(st)[1]


#Coordinates of place to predict (ASKIM)
a <- 57.6339 # Latitude
b <- 11.9399 # Longitude
po_i = c(a,b)
date_i = "1996-02-23"
times_i = c("04:00:00", "06:00:00", "08:00:00", "10:00:00", 
            "12:00:00", "14:00:00", "16:00:00", "18:00:00", 
            "20:00:00", "22:00:00", "24:00:00") #11 different times of interest

n = length(times_i)

#Remove posterior data
filtered_st <- st[st$date >= "1941-08-31" & st$date <= "1996-02-23",]

#Table for coordinates
cnames1 = c("Latitude", "Longitude")
coord = c(a,b)
content = cbind(cnames1, coord)

kbl(content, caption = "Coordinates for place of interest", col.names = c("Askim", "Coordinates")) %>%
  kable_styling(latex_options = "hold_position", position = "left")

####KERNEL FOR DISTANCE######

# Weighing factor for distance
h_dist <- 250000 #meter -> 250km

# Kernel 1, Function for calculating the distance from a station to a point of interest
dist_kernel = function(poi, data, h) {
  dist <- distHaversine(poi, data.frame(data$latitude, data$longitude)) #returns in meter
  u <- dist/h
  return (exp(-u^2))
}


####KERNEL FOR DATE######
# Weighing factor for date
h_date <- 20

#Kernel 2, distance between the day a temperature measurement was made and the day of interest
day_kernel = function(doi, data, h) {
  diff_tot <- (as.numeric(as.Date(data$date) - as.Date(doi), unit="days")) #difference in unit days 
  diffdays <- diff_tot%%365 #take the year in consideration
  u <- diffdays/h
  return (exp(-u^2)) #kernel
}

####KERNEL FOR TIME######
# Weighing factor for time
h_time <- 4

#Kernel 3, distance between the hour of the day a temperature measurement was made and the hour of interest
hour_kernel = function(toi, data, h) {
  diff_sec <- difftime(strptime(data$time , format = "%H:%M:%S"),  #time differences in seconds
                       strptime(toi , format = "%H:%M:%S"))        #date-time conversion functions to and from character
  diff_hour <- as.numeric(diff_sec/(3600))                              #from sec to  hour
  u <- diff_hour/h
  return (exp(-u^2)) #kernel
}


# Calculate the sum of the three gaussian kernels


#Function for calculate the predicted temperature by the sum of kernels at a certain time 
calcSum = function(data, poi, doi, hoi) {
  dist_k = dist_kernel(poi, filtered_st, h_dist)
  date_k = day_kernel(doi, filtered_st, h_date)
  hour_k = hour_kernel(hoi, filtered_st, h_time)
  sum_k <- dist_k+date_k+hour_k #summing up the kernel values
  
  #Prediction of the temp for the 11 kernel values with (Gaussian) Kernel Regression, weighted kernel for the observations
  sum_temp = sum(sum_k * filtered_st$air_temperature)/sum(sum_k) 
  return(sum_temp)
}
pred_temp_sum = c()
#Looping through time in times of interest and predict the temperature
for (i in 1:n) {
  pred_temp_sum[i] = calcSum(filtered_st, po_i, date_i, times_i[i])
}

#Create table with the time corresponding and temperatures
table <- cbind(times_i, pred_temp_sum)
kbl(table, caption = "Temperatures for sum kernel", col.names = c("Time", "Temp")) %>%
  kable_styling(latex_options = "hold_position", position = "left")


#### FINDING SMOOTHING H-VALUES, plot kernel value as function of 
#### days, distance and

#Plotting weights of the distance kernel
dist_plot = function(dist_kernel, h){
  u = dist/h
  plot(dist, exp(-u^2), type="l", main="Weights for distance, h = 250000", xlab="Distance", ylab = "Kernel")
}

#Sequence generator
dist = seq(0, 150000, 1)
dist_plot(dist, h_dist)

#Plotting weights of the date kernel
date_plot = function(date_kernel, h){
  u = date/h
  plot(date, exp(-u^2), type="l", main="Weights for date, h = 20", xlab="Days", ylab = "Kernel")
}

date = seq(0,100,1)
date_plot(date, h_date)
#----------------------------------------------------------------------------------------------------

#Plotting weights of the hour kernel
hour_plot = function(times, h){
  u = times/h
  plot(time, exp(-u^2), type="l", main="Weights for hour, h = 4", xlab="Hours", ylab = "Kernel")
}

time = seq(0,12,1)
hour_plot(time, h_time)


### COMBINE KERNELS BY MULTIPLICATION INSTEAD OF SUM ###

#Multiplying the kernels instead of summing them up

#Function for calculate the predicted temperature by the product of kernels at a certain time 
calcProd = function(data, poi, doi, hoi) {
  dist_k = dist_kernel(poi, data, h_dist)
  date_k = day_kernel(doi, data, h_date)
  hour_k = hour_kernel(hoi, data, h_time)
  mult_k = dist_k*date_k*hour_k
  
  #Calculate the predicted 11 kernel values for temperature with (Gaussian) Kernel Regression
  mult_temp = sum(mult_k * filtered_st$air_temperature)/sum(mult_k)
  return(mult_temp)
}

pred_temp_prod = c()
#Prediction of temperature for times of interest
for(i in 1:n){
  pred_temp_prod[i] = calcProd(filtered_st, po_i, date_i, times_i[i])
}

#Table for values
table2 <- cbind(times_i, pred_temp_prod)
kbl(table2, caption = "Temperatures for product kernel", col.names = c("Time", "Temp")) %>%
  kable_styling(latex_options = "hold_position", position = "left")

#Comparison between the summed kernels and the multiplied kernels
#Plots of the predicted temperatures
hours = seq(4, 24, 2)
plot(hours, pred_temp_sum, type="o", main="Plot for summed kernels at [1996-02-23]", ylab="Temp", xlab="Hour")
plot(hours, pred_temp_prod, type="o", main="Plot for multiplied kernels at [1996-02-23]", ylab="Temp", xlab="Hour")