library(datasets)
library(mvtnorm)
library(nnet)
library(MASS) 
library(dplyr) 
library(knitr)
library(kableExtra)
library(ggplot2)
library(matlib)

#Devide data from image(?) Cool i guess 
data(iris)
summary(iris)
#1 Make 
qplot(Sepal.Length, Sepal.Width, data=iris, color = Species, main = "Scatterplot Iris")

#2 Use simple R-functions to implement LDA between the three species(based on sepal w vs l)
# a, Compute mean, covariance matrix and prior probabilities per class
# Get different species from dataset
setosas = filter(iris, Species == "setosa") #Filter iris data from setosas
setosas = setosas[,1:2] #ONly use the first two columns
versicolor = filter(iris, Species == "versicolor") #Filter iris data from versicolor
versicolor = versicolor[,1:2] 
virginica = filter(iris, Species == "virginica") #Filter iris data from virginica
virginica  = virginica [,1:2] 

# Create one vector with the labels and one with the values. 
names <- c("Setosa length", "Setosa width", "Versicolor length", "Versicolor width", "Virginica length", "Virginica width") 
means <- c(mean(setosas$Sepal.Length),mean(setosas$Sepal.Width),mean(versicolor$Sepal.Length),mean(versicolor$Sepal.Width),mean(virginica$Sepal.Length),mean(virginica$Sepal.Width))

#Show the calculated means 
sepal.names = setNames(names, means)
sepal.names

#Calculate covariance
cov.setosa = cov(setosas)
cov.versicolor = cov(versicolor)
cov.virginica = cov(virginica)
cov.setosa
cov.versicolor
cov.virginica
rm(names, means)
#Prior probabilities - Prior known prob, the number of samples in each group compared to the total, 1/3 in this case
#Count the number of rows for the different classes of irises
sets = count(iris, Species)
#Set names
names <- c("Setosa", "Versicolor", "Virginica")
# calculate the prior probabilites for each species and add to vector. 
pis <- matrix(c("Setosa", "Versicolor", "Virginica", sets$n[1]/count(iris), sets$n[2]/count(iris), sets$n[3]/count(iris)),nrow=3, ncol=2)
pis

# b, Compute overall cov.matrix
# We calculate the pooled covariance by adding the three cov.matrixes together(with regards to prior probability i guess)
cov.pooled = (cov.setosa + cov.versicolor + cov.virginica)/3
inv.pooled.cov = inv(cov.pooled)
cov.pooled
inv.pooled.cov
# c, report probibalistic model for the LDA
#The prob. model for LDA is x|((y = Ci), ui, sum) ~ N(ui, sum)
# Ci is the class of the observation, ui is the mean for the same class and sigma is the summed variance for all classes

# d, Compute discriminant functions for each class
#See report

# e, compute equations of decision bounduries between classes

# Calculate u_hat
uHat.setosa <- matrix(c(sum(setosas$Sepal.Length)/50, sum(setosas$Sepal.Width)/50))
uHat.setosa
uHat.versicolor <- matrix(c(sum(versicolor$Sepal.Length)/50, sum(versicolor$Sepal.Width)/50))
uHat.versicolor
uHat.virginica <- matrix(c(sum(virginica$Sepal.Length)/50, sum(virginica$Sepal.Width)/50))
uHat.virginica

#calculate w
w.setosa <- inv.pooled.cov%*%uHat.setosa
w.versicolor <- inv.pooled.cov%*%uHat.versicolor
w.virginica <- inv.pooled.cov%*%uHat.virginica


t.uHat.versicolor <- t(uHat.versicolor)
t.uHat.virginica <- t(uHat.virginica)
t.uHat.setosa <- t(uHat.setosa)

#calculate w0
w0.setosa <- (-0.5)*t.uHat.setosa%*%inv.pooled.cov%*%uHat.setosa + log(1/3)
w0.versicolor <- (-0.5)*t.uHat.versicolor%*%inv.pooled.cov%*%uHat.versicolor + log(1/3)
w0.virginica <- (-0.5)*t.uHat.virginica%*%inv.pooled.cov%*%uHat.virginica + log(1/3)

#calculate boundury between classes
# Calculate decision boundary between setosa and versicolor 
w.setosa - w.versicolor
w0.setosa - w0.versicolor

# Calculate decision boundary between setosa and virginica
w.setosa - w.virginica
w0.setosa - w0.virginica

# Calculate decision boundary between virginica and versicolor
w.versicolor - w.virginica
w0.versicolor - w0.virginica

#3 Use discriminant function from 2d, to predict species from original data.
# Make predictions for the iris dataset based on our discriminant functions.
# Array to store sepal length, width, prediction and correct class.
predictions <- data.frame("Sepal.Length" = double(), "Sepal.Width" = double(), "pred" = factor(), "true_class" = factor())

for (data in 1:nrow(iris)) {
  # X^T of the sepal inputs. 
  x_t <- matrix(c(iris$Sepal.Length[data], iris$Sepal.Width[data]),nrow = 1, ncol = 2)
  
  # delta value for setosa. 
  delta_setosa <- (x_t %*% inv.pooled.cov %*% uHat.setosa - (1/2)*t.uHat.setosa%*%inv.pooled.cov%*%uHat.setosa + log(1/3))
  
  # delta value for versicolor.
  delta_versicolor <- (x_t %*% inv.pooled.cov %*% uHat.versicolor - (1/2)*t.uHat.versicolor%*%inv.pooled.cov%*%uHat.versicolor + log(1/3))
  
  # delta value for virginica. 
  delta_virginica <- (x_t %*% inv.pooled.cov %*% uHat.virginica - (1/2)*t.uHat.virginica%*%inv.pooled.cov%*%uHat.virginica + log(1/3))  
  
  # Choose the highest delta and predict to that species. 
  # If all three deltas are the same classify as Setosa. 
  if (delta_setosa >= delta_versicolor && delta_setosa >= delta_virginica) {
    # Predict Setosa.
    new_pred <- as.factor("setosa")
  } else if (delta_virginica >= delta_versicolor) {
    # Predict virginica
    new_pred <- as.factor("virginica")
  } else {
    # Predict versicolor
    new_pred <- as.factor("versicolor")
  }
  
  # Store in the predictions table. 
  predictions <- rbind(predictions, data.frame("Sepal.Length"=iris$Sepal.Length[data],"Sepal.Width"= iris$Sepal.Width[data],"pred"=new_pred,"true_class"= iris$Species[data]))

}

# Plot the predictions reviewed from the discriminant functions. 
qplot(Sepal.Length, Sepal.Width, data=predictions, colour=pred)

# Function to calculate the misclassification error.
missclass=function(X,X1){
  n=length(X)
  return(1-sum(diag(table(X,X1)))/n)
}

# Calculate misclassification rate for predictions using the discriminant functions.
mc.discPredictions <- missclass(predictions$true_class, predictions$pred)
mc.discPredictions
# Import the mass-library to get LDA functionality. 
library(MASS)

# Train a model using the LDA function. 
model.lda <- lda(Species ~ Sepal.Length + Sepal.Width, data = iris)

# Make predictions on the iris set using the trained model. 
pred.lda <- predict(model.lda, iris[,1:2])$class

# Calculate the misclassification rate. 
mc.ldaPred <- missclass(iris$Species, pred.lda)
mc.ldaPred

#4, Use model from 2c to generate new data with same number of total cases
library(mvtnorm)

generated.iris <- data.frame("Sepal.Length"= double(), "Sepal.Width"=double(), "Species"=factor())

# Generate new observations. 
# Once for every data in the iris data set. 
for (i in 1:nrow(iris)) {
  # Sample one random observation from the iris data set. 
  # Use replacement. 
  data <- sample(iris$Species, 1, replace = TRUE)
  
  # Decide what species we have sampled. 
  if (data=="setosa") {
    new.data <- rmvnorm(n = 1, mean = uHat.setosa, sigma = cov.setosa)
  } else if (data=="versicolor") {
    new.data <- rmvnorm(n = 1, mean = uHat.versicolor, sigma = cov.versicolor)
  } else {
    new.data <- rmvnorm(n = 1, mean = uHat.virginica, sigma = cov.virginica)
  }
  
  # Add new data to generated data list. 
  generated.iris <- rbind(generated.iris, data.frame("Sepal.Length"=new.data[1], "Sepal.Width"=new.data[2],"Species"=data))
}

# 5, Step 3 but using LOGISTIC REGRESSION instead.

library(nnet)

# Train the model.
iris.mu <- multinom(Species ~ Sepal.Length + Sepal.Width, iris)
# The predicted species. 
lr.pred.species <- predict(iris.mu)

# Data frame to store our predictions. 
lr.pred <- iris[,1:2]
# Add predicted species. 
lr.pred <- cbind(lr.pred, data.frame("Prediction"=lr.pred.species, "Species"=iris$Species))

# Misclassification error
mc.lr.pred <- missclass(lr.pred$Species, lr.pred$Prediction)
mc.lr.pred
# Plot the predictions made using logistic regression. 
qplot(Sepal.Length, Sepal.Width, data=lr.pred, colour=Prediction)

# Plot the generated observations. 
qplot(Sepal.Length, Sepal.Width, data=generated.iris, colour=Species)





