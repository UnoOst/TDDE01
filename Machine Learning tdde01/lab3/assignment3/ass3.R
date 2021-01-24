# Import neuralnet package. 
library(neuralnet)

# Sample 500 points. 
set.seed(12345) # Set seed to have reproducible results.
data1 <- runif(500,0,10)
data <- data.frame("obs"= data1, "sin" = sin(data1))

# Split data1 into training and test sets. 
n <- dim(data)[1]
set.seed(12345) # Seed for reproducible results. 
id <- sample(1:n, 25) # Take sample of 25 observations. 
# Create train and test set. 
train <- data[id,] 
test <- data[-id,]

rm(n, id, data1) # Remove unnecessary variables. 
runif()

#1 Train NN on sin, interval [0..10], 500 observations
library(ggplot2) # To have good looking plots. 
library(grid)
library(gridExtra)

# Plot training data
ggplot(train,aes(x=obs, y=sin)) +
  labs(x="x", y="sin(x)", title = "Training data") +
  geom_line(linetype="dashed", color="blue") -> p1

# Plot test data 
ggplot(test, aes(x= obs, y=sin)) +
  labs(x="x", y="sin(x)", title = "Test data") +
  geom_line(linetype="dashed", color="darkred") -> p2

grid.arrange(p1,p2, ncol=2)


## Fit model using neuralnet(). Two layers, 4 resp, 3 hidden units, random start weights [-1,1]
## Only one layer and 6 units, no good results

# Initialize weights in interval [-1,1]
set.seed(12345)
weight <- runif(7,-1,1)

# Fit the neural network using the train set. 
set.seed(12345) # Without seed we seem to get random results. 
fit.neural <- neuralnet(sin ~ obs, data= train, hidden = c(4,3), startweights = weight)

# Predict using the fit. 
prediction <- predict(fit.neural, test)

# plot the prediction. 
ggplot(test, aes(x=obs)) + 
  geom_line(aes(y = prediction, color="darkred")) + 
  labs(x="x", y =" Predicted sin(x)", title="Predictions on test set") + 
  geom_line(aes(y=sin, color="steelblue"), linetype = "dashed") + 
  scale_color_discrete(name = "", labels = c("Predictions", "True sin values"))

# 2, same but with interval [0..20]
# Sample 500 points. 
set.seed(12345) # Set seed to have reproducible results.
x <- runif(200,0,20)
data2 <- data.frame("obs"= x, "sin" = sin(x))

rm(x) # Remove unnecessary variables. 

# Predict using the model from part 1. 
pred2 <- predict(fit.neural, data2)

# plot the predictions on data 2
ggplot(data2, aes(x=obs)) + 
  geom_line(aes(y=pred2, colour="darkred")) + 
  labs(x = "x", y = "sin(x)", title="Predictions on new data") + 
  geom_line(aes(y=sin, colour="steelblue"),linetype="dashed") + 
  scale_color_discrete(name = "", labels = c("Predictions", "True sin values"))

# 3

# Sample 500 points. 
set.seed(12345) # Set seed to have reproducible results.
data1 <- runif(500,0,10)
data3 <- data.frame("obs"= data1, "sin" = sin(data1))

rm(data1) # Remove unnecessary variables. 

# Weights again 
set.seed(12345)
weight <- runif(7, -1, 1)
# Train NN model on entire data 3 set. 
# Use same number of hidden variables as part 1. 
# Use same random weights as in part 1. 
# Fit x (observations) based on it's corresponding sin value. 
set.seed(12345)
fit.sin_neural <- neuralnet(obs ~ sin, data=data3, hidden=c(6), startweights = weight)

# Predict x using fitted model. 
pred.x <- predict(fit.sin_neural, data3)

# Plot the predictions. 
ggplot(data3, aes(y=sin)) + 
  geom_line(aes(x=pred.x, colour="darkred")) + 
  labs(x = "x", y = "sin(x)", title="Predictions on new data") + 
  geom_line(aes(x=obs, colour="steelblue"),linetype="dashed") + 
  scale_color_discrete(name = "", labels = c("Predictions", "True sin values"))
