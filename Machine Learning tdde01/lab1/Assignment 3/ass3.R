
library(ggplot2)
library(glmnet)

##Assignment 3

#This part deals with linear regression using the LASSO method

## Dividing the data
#First we need to divide the data into training 
#and testing data, a randomly selected 50/50 split

tecator = read.csv("tecator.csv")
n <- dim(tecator)[1]
set.seed(12345)
id = sample(1:n, floor(n*0.5))
train <- tecator[-id,]
test <- tecator[id,]


# task 1
#make a linear regression model for fat where absorbency char. are features.

fat_model = lm((Fat)~., data = train[,2:102])

lm_yhat_train = predict(fat_model, data = train[,2:101])
lm_yhat_test = predict(fat_model, data = test[,2:101])

fat_train_c = train$Fat
fat_test_c = test$Fat

MSE = function(pred, true) {
  return(sum((true - pred)^2)/length(true))
}

err_train = MSE(lm_yhat_train, fat_train_c)
err_train
err_test = MSE(lm_yhat_test, fat_test_c)
err_test

#(fat_model)
#plot(fat_model)

#2 LASSO regression - report objective function
# W_hat = argmin ( sum(True_fat - Pred_fat)^2 + lambda*sum(abs(w)))
#w_hat(LASSO) = argmin (sum[i = 1..N](y(i)- w(0) - w1())x(1,j) - ... - w(p)X(p)j)^2) + lambda*sum[1..j](abs(w(i)))

#3 Fit LASSO model to training data. 
# Make plot of how reg.coeffs depend on log(lambda)

covariate = train[,2:101]
response = train[,102]

fat_lasso = glmnet(as.matrix(covariate), response, alpha = 1, family = "gaussian")

plot(fat_lasso, xvar = "lambda", label = TRUE)

#4 Plot how DF depend on lambda
plot(fat_lasso[["lambda"]], fat_lasso[["df"]])



#Finding optimal lambda
optimal_lambda <- fat_lasso_DF$lambda.min
optimal_lambda

#5 Do task #3 but for RIDGE regression instead
fat_ridge = glmnet(as.matrix(covariate), response, alpha = 0, family = "gaussian")

plot(fat_ridge, xvar = "lambda", label = TRUE, main = "Active coeffs for different Lambda Ridge")

#6 Do CV for LASSO model, comment on how CV-score changes with log(lambda). 
# Is it stat.sign. better than log(lambda) = -2
# Also make a scatterplot of the original test values vs pred. test values. 
#Plot graph showing how MSE depends on Lambda with crossvalidation
fat_lasso_DF = cv.glmnet(as.matrix(covariate), response, alpha = 1, family = "gaussian")
plot(fat_lasso_DF, xvar = "lambda", label = TRUE)
coef(fat_lasso_DF, s = "lambda.min")
print(fat_lasso_DF$lambda.min)

yhat = predict(fat_lasso_DF, newx = as.matrix(test[,2:101]), s="lambda.min")
plot(yhat, test$Fat,  col = "orange", main = "True vs pred Fat values", xlab = "True value")
error_test = MSE(yhat, test$Fat)
error_test
#7 Create new target values from true and predicted data 

fat_lasso_DF[["cvsd"]]
sd = sqrt(error_test)
sd
generated_test = rnorm(107, yhat, sd)
plot(generated_test, test$Fat, col = "orange")
points(yhat, test$Fat, col = "blue")
