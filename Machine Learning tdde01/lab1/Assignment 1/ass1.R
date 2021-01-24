library(cvTools)
library(kknn)
library(readr)

#1 Divide the data
opt= read.csv("optdigits.csv")
n <- dim(opt)[1]
set.seed(12345)
id=sample(1:n, floor(n*0.5))
train=opt[id,]
id1=setdiff(1:n, id)
set.seed(12345)
id2=sample(id1, floor(n*0.25))
valid=opt[id2,]
id3=setdiff(id1, id2)
test=opt[id3,]

#2 Fit 30-nearest neighbor
# Make the last column into factor (x65)
opt$X0.26 = as.factor(opt$X0.26)
train$X0.26 = as.factor(train$X0.26)
test$X0.26 = as.factor(test$X0.26)
valid$X0.26 = as.factor(valid$X0.26)

# Create model and predict for training and test data
kknn_test = kknn(formula = X0.26 ~., train = train, test = test, k = 30, kernel = "rectangular")
kknn_pred_test = predict(kknn_test)

kknn_train=kknn(X0.26~., train=train, test=train, k = 30, kernel = "rectangular")
kknn_pred_train = predict(kknn_train)

#Create confusion matrixes
cm_test = table(test$X0.26 , kknn_pred_test) 
cm_train = table(train$X0.26 , kknn_pred_train)
cm_test
cm_train

# Number sum matrix - diagonal (answer in %)
missclass=function(X,X1){
  n=length(X)
  return(1-sum(diag(table(X,X1)))/n)
}
mc_kknn.train <- missclass(train$X0.26, kknn_pred_train)
mc_kknn.train
mc.kknn.test <- missclass(test$X0.26, kknn_pred_test)
mc.kknn.test


#3 find any 2 cases of digit "8" that are easiest to classify and 3 that are hardest to classify.
#visualize using heatmap(8x8)
#Find 2 easiest cases (closest distance from kknn)

# First we add the probabilities for the test data to a table 
# with all the test data. We do this so we can sort the data in regards
# to the probabilites. 

test.prob <- test

prob_8 <- kknn_test$prob[,9]

test.prob$prob <- prob_8
rm(prob_8)

test.probb <- test
prob_8 <- kknn_test$prob[,9]

test.probb$prob <- prob_8

library(dplyr)
test.probb <- filter(test.probb, X0.26 ==8)
options(max.print = 100000)
test.probb

A=data.matrix(test.probb[2, 1:64]) 
AA=matrix(A, nrow=8, ncol = 8)
B=data.matrix(test.probb[5, 1:64]) 
BB=matrix(B, nrow=8, ncol = 8)

heatmap(t(AA), Rowv=NA, Colv=NA)
heatmap(t(BB), Rowv=NA, Colv=NA)

#Three cases of digit 8 that were difficult to classify (found in the probability data in kknn): row number 74, 69 and 89
C = data.matrix(test.probb[78, 1:64]) 
CC = matrix(C, nrow=8, ncol=8)
E = data.matrix(test.probb[31, 1:64]) 
EE = matrix(E, nrow=8, ncol=8)
G = data.matrix(test.probb[74, 1:64]) 
GG = matrix(G, nrow=8, ncol=8)
heatmap(t(GG), Rowv=NA, Colv=NA)
heatmap(t(CC), Rowv=NA, Colv=NA)
heatmap(t(EE), Rowv=NA, Colv=NA)

# Filter out so we only have rows with X65 = 8. 
# (Eight is the target value). 


test.prob <- filter(test.prob, X0.26 == 8)

test.prob <- test.prob[order(test.prob$prob),]

worst3 <- head(test.prob, n =3)
worst3
best2 <- tail(test.prob, n=2)
best2

rowToMatrix=function(V){
  A = matrix(,8,8)
  x = 1 
  for(row in 1:8) {
    for(col in 1:8) {
      A[row,col] <- (V[x])
      x <- x+1
    }
  }
  rm(x)
  return(A)
}

bestRow1 <- rowToMatrix(as.numeric(best2[1,]))
bestRow2 <- rowToMatrix(as.numeric(best2[2,]))

worstRow1 <- rowToMatrix(as.numeric(worst3[1,]))
worstRow2 <- rowToMatrix(as.numeric(worst3[2,]))
worstRow3 <- rowToMatrix(as.numeric(worst3[3,]))

heatmap(bestRow1, Colv=NA, Rowv=NA, main="Best predicted eight.")
heatmap(bestRow2, Colv=NA, Rowv=NA, main="2nd best predicted eight.")

heatmap(worstRow1, Colv=NA, Rowv=NA, main="Worst Predicted 8.")
heatmap(worstRow2, Colv=NA, Rowv=NA, main="2nd worst predicted 8.")
heatmap(worstRow3, Colv=NA, Rowv=NA, main="3rd worst predicted 8.")

# 4 KNN fit with different K-values(# of neighbors), from 1..30

# Firstly, create a new dataset to store the missclassification rate. 
k <- c()
Rate_misclass <- c()
misClass_train <- data.frame(k, Rate_misclass)
misClass_valid <- data.frame(k, Rate_misclass)

# Do knn fitting with k-values ranging from 1..30

for (i in 1:30) {
  #Train for different k-values (i)
  k <- train.kknn(formula = X0.26 ~., data = train, ks = i, kernel = "rectangular")
  # Predictions for the training dataset
  pred.k = predict(k, train)
  # Calculate the MC rate
  mc = missclass(train$X0.26, pred.k)
  misClass_train <- rbind(misClass_train, c(i, mc))
  
  # Predictions for the validation set
  pred.k = predict(k, valid)
  # Calc MC rate
  mc = missclass(valid$X0.26, pred.k)
  misClass_valid = rbind(misClass_valid, c(i, mc))
  
}
#Remove unnecessary variables
rm(mc); rm(pred.k); rm(k)

plot(misClass_train$X1, misClass_train$X0, type="l", ylim = c (0,0.08), 
     xlab = "k", ylab = "Misclassification Rate", 
     main = "Misclassification Rates for different K", col = "orange")
points(misClass_valid$X1, misClass_valid$X0.0293193717277487, type="l", 
       col = "blue")
#Estimate MC_error for optimal K (7 in our case) compare for training and Validation set. 
#Do knn fitting for optimal K-value
k <- train.kknn(formula = X0.26 ~., data = train, ks = 4, kernel = "rectangular")
#prediction for testset
pred.k <- predict(k, test)
#Calc MC-rate
mc_K_4 <- missclass(test$X0.26, pred.k)
#Create table/matrix to show MC-rates for different datasets with k=4
matrix(c(misClass_train[7,2], misClass_valid[7,2], mc_K_4), 
       nrow = 1, ncol = 3, dimnames = list("misclassificationrate", c("Training set", "Validation set", "Test set")))

#5 Cross-entropy calculation of empirical risk

k <- c()
C_E = c()
C_E_Val = data.frame(k, C_E)

# Fit the model for different K-values
for (i in 1:30) {
  k <- kknn(formula = X0.26 ~., train = train, test = valid, k = i, kernel = "rectangular")
  
  # Calculating of the cross entropy 
  x <- 0
  for (j in 1:nrow(valid)) {
    #Get prop of correct classification for each observation
    p_hat = k$prob[j, as.numeric(valid$X0.26[j])]
    x = x + log(p_hat+(1e-15))

  }
  C_E_Val <- rbind(C_E_Val, c(i, (-x)))
}

plot(C_E_Val, ylab="cross entropy", xlab="k", main="cross entropy for the validation set")

