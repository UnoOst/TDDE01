# Import libraries
library(cvTools)
library(MLmetrics)
library(kknn)
library(DMwR)


# 1 Write down the probabilistic model as a bayesian model

#P(motor_UPDRS|w, sigma) = N(motor_UPDRS|w0 + wX, sigma^2I)
# w ~ N(0, sigma^2*I/lambda)

#2 Scale the data to training and test set (60/40)

# Read the data 
parkinsons_data <- read.csv("../Assignment 2/parkinsons.csv")

#Scale the data
parkinsons_data = scale(parkinsons_data)

#Divide into training and test set
n = dim(parkinsons_data)[1]
set.seed(12345)
id = sample(1:n, floor(n*0.6))
training = parkinsons_data[id, ]
test = parkinsons_data[-id, ]

#3 Implement 4 functions without using packages or help.

#LOGLIKELIHOOD
loglike = function(data, w, sigma) {
  
  n = dim(data)[1] #Total number of observations
  Y = data[,5] #The iifth column in the dataset is the real motor_UPDRS values
  X = data[,7:22] #The features in the column 7-22, with relevant parameters.
  sigma_sq = sigma^2
  result = -n/2*log(2*pi) - n/2*log(sigma_sq)-t(Y-X%*%w)%*%(Y-X%*%w)/(2*sigma_sq)
  return(result)
}

#RIDGE
ridge = function(data, w, lambda) {
  sigma = w[1]#The first value in w is the sigma
  w = w[-1] #The remaining 16 values is the w-vector
  result = -loglike(data, w, sigma) + lambda*sum(w^2)
  return(result)
}

#RIDGE_OPT
ridge_opt = function(data, lambda) {
  w = rep(1, 17)
  opt = optim(par = w, fn = ridge, lambda = lambda, data = data, method = "BFGS")
  return(opt)
}

#DF
df = function (data, lambda) {
  x = data[,7:22]
  P <- x%*%solve(t(x)%*%x+lambda*diag(dim(x)[2]))%*%t(x)
  return(sum(diag(P)))
}
lambda1 = df(training, 1)
lambda1
lambda100 = df(training, 100)
lambda100
lambda1000 = df(training, 1000)
lambda1000
#4. Find optimal sigma and w params and then compute MSE for both training and test set.
#LAMBDA = 1
#LAMBDA = 100
#LAMBDA = 1000

ridge_1_train = ridge_opt(training, 1)$par[1]
ridge_1_test = ridge_opt(test, 1)$par[1]

ridge_100_train = ridge_opt(training, 100)$par[1]
ridge_100_test = ridge_opt(test, 100)$par[1]

ridge_1000_train = ridge_opt(training, 1000)$par[1]
ridge_1000_test = ridge_opt(test, 1000)$par[1]

w_1_train <- ridge_opt(training, 1)$par[2:17]
w_1_test <- ridge_opt(test, 1)$par[2:17]

w_2_train <- ridge_opt(training, 100)$par[2:17]
w_2_test <- ridge_opt(test, 100)$par[2:17]

w_3_train <- ridge_opt(training, 1000)$par[2:17]
w_3_test <- ridge_opt(test, 1000)$par[2:17]

Y_train = training[,5]
Y_test = test[,5]

X_train = training[,7:22]
X_test = test[,7:22]

predic_Y_1_train <- X_train%*%w_1_train
predic_Y_1_test <- X_test%*%w_1_test

predic_Y_2_train <- X_train%*%w_2_train
predic_Y_2_test <- X_test%*%w_2_train

predic_Y_3_train <- X_train%*%w_3_train
predic_Y_3_test <- X_test%*%w_3_test


MSE_1_train <- (1/n)*(sum(predic_Y_1_train - Y_train)^2)
MSE_1_train
MSE_1_test <- (1/n)*(sum(predic_Y_1_test - Y_test)^2)
MSE_1_test
MSE_2_train <- (1/n)*(sum(predic_Y_2_train - Y_train)^2)
MSE_2_train
MSE_2_test <- (1/n)*(sum(predic_Y_2_test - Y_test)^2)
MSE_2_test
MSE_3_train <- (1/n)*(sum(predic_Y_3_train -  Y_train)^2)
MSE_3_train
MSE_3_test <- (1/n)*(sum(predic_Y_3_test - Y_test)^2)
MSE_3_test

#5 AIC, Akaike Information Criterion
AIC_1_train <- 2*df(training, 1) - 2*loglike(training, w_1_train, sigma_1_train) 
AIC_1_test <- 2*df(test, 1) - 2*loglike(test, w_1_test, sigma_1_test) 
AIC_1_train
AIC_1_test
AIC_100_train <- 2*df(training, 100) - 2*loglike(training, w_2_train, sigma_2_train) 
AIC_100_test <- 2*df(test, 100) - 2*loglike(test, w_2_test, sigma_2_test) 
AIC_100_train
AIC_100_test
AIC_1000_train <- 2*df(training, 1000) - 2*loglike(training, w_3_train, sigma_3_train) 
AIC_1000_test <- 2*df(test, 1000) - 2*loglike(test, w_3_test, sigma_3_test) 
AIC_1000_train
AIC_1000_test
