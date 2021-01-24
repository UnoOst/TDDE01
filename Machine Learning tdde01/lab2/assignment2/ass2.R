# Libraries
library(tree)
library(rpart)
library(MASS)
library(e1071)

#1 Devide data, exlude parameter "Duration". 40/30/30
data  = read.csv2("bank-full.csv", stringsAsFactors = T)
#Remove duration var
data = subset(data, select = -c(duration))

#Divide data to 40/30/30
n = dim(data)[1]
set.seed(12345)
id = sample(1:n, floor(n*0.4))
train = data[id,]
id1 = setdiff(1:n, id)
set.seed(12345)
id2 = sample(id1, floor(n*0.3))
valid = data[id2,]
id3 = setdiff(id1, id2)
test = data[id3,]

#convert y to factor
data$y = as.factor(data$y)

#2 Fit decision tree to train data so that you can change settings
# default settings
tree_default = tree(as.factor(y)~., data=train) 

#min node size = 7000
tree_node = tree(as.factor(y)~., data=train, minsize=7000)

#min diviance = 0.0005 
tree_dev = tree(as.factor(y)~., data=train, mindev=0.0005)

# Misclassification error function
misclass=function(X,X1){
  n=length(X)
  return(1-sum(diag(table(X,X1)))/n)
}


# Predict on the training data
Y_tree_default=predict(tree_default, train, type = "class")
Y_tree_node=predict(tree_node, train, type = "class")
Y_tree_dev=predict(tree_dev, train, type = "class")

# Predict on the validation data. 
Y_tree_default_vali=predict(tree_default, valid, type = "class")
Y_tree_node_vali=predict(tree_node, valid, type = "class")
Y_tree_dev_vali=predict(tree_dev, valid, type = "class")

# Calculate the misclassification errors. 
mc_default <- misclass(train$y, Y_tree_default)
mc_node <- misclass(train$y, Y_tree_node)
mc_dev <- misclass(train$y, Y_tree_dev)

mc_default_vali <- misclass(valid$y, Y_tree_default_vali)
mc_node_vali <- misclass(valid$y, Y_tree_node_vali)
mc_dev_vali <- misclass(valid$y, Y_tree_dev_vali)

df <- data.frame(c(mc_default, mc_node, mc_dev), c(mc_default_vali, mc_node_vali, mc_dev_vali))
row.names(df) <- c("Default fit", "Min node size", "min Deviance")
colnames(df) <- c("Training set", "Validation set")
df

# 3, Choose optimal tree depth(up to 50) and present
# Graph of dependence of deviance on numb. of leaves
# Optimal #of_leaves and most important vars for decisionmaking
# Est. confusion matrix & misclass rate on test-set.

# Create to vectors to store results in
train_score = rep(0,50)
test_score = rep(0,50)

for (i in 2:50) { #For loop iterate through different number of leaves
  
  # Prune best performing tree from #2, find 
  pruned_tree = prune.tree(tree_dev, newdata=train, best = i)
  
  # Predict on validation set
  pred = predict(pruned_tree, newdata = valid, type = "tree")
  
  #Store results
  train_score[i] = deviance(pruned_tree)
  test_score[i] = deviance(pred)
  
  }

#Now we want to plot the graph for dependence of deviance for number of leaves (2:50)

plot(2:50, train_score[2:50], type = "b", col = "orange", ylim = c(8000, 12450),
main = "Trainging & Validation Deviance", xlab = "Nr of leaves", ylab = "Deviance")

points(2:50, test_score[2:50], type = "b", col = "blue")

plot(2:50, test_score[2:50], type = "b", col = "red", main = "Validation Dataset Deviance", xlab = "Nr of leaves", ylab = "Deviance")

opt_leaves = min(test_score[2:50], na.rm = "TRUE")
opt_leaves
for (i in 1:50) {
  print(i)
  print(test_score[i])  
}

pruned_tree = prune.tree(tree_dev, best = 22)
plot(pruned_tree)
text(pruned_tree, pretty = 1)

#Predict using pruned tree with 22 leaves
fit = predict(pruned_tree, newdata = test, type = "class")
#make and print confusion matrix
table(test$y, fit)
# Calc. missclass-rate
mc_opt_nr_leaves = misclass(test$y, fit)
mc_opt_nr_leaves

#4, make Decision tree using loss matirx
#Report confusion matrix and compare results

pred_tree_loss_matrix = predict(tree_dev, newdata = test, type = "vector")

#Create DF to store predictions
pred_lm = data.frame("pred" = factor())

#Loop for making predictions.
# To class "no", has to have 5x larger prob than "yes"
for (i in 1:nrow(pred_tree_loss_matrix)) {
  if (pred_tree_loss_matrix[i,2]*5 > pred_tree_loss_matrix[i,1]) {
    #classify as no
    pred_lm = rbind(pred_lm, data.frame("pred" = as.factor("yes")))
  } else {
    pred_lm = rbind(pred_lm, data.frame("pred" = as.factor("no")))
  }
}
pred_lm
#Confusion Matrix
table(test$y, pred_lm$pred)

#misclass rate (should increase)

mc_loss_matrix_tree = misclass(test$y, pred_lm$pred)
mc_loss_matrix_tree


#5, Use optimal tree & Naive Bayes to classify data using given model
# Compute TPR, FPR and plot a ROC - curve (Reciever Operation Char - curve)

#Function to calc TPR
# Function to calculate TPR 
calculate_tpr = function(T, F) {
  # No predictions need to be factor 1
  ctable <- table(T,F)
  N <- sum(ctable[2,])
  tp <- ctable[2,2]
  return (tp/N)
}


#Function to calc FPR
# Function to calculate TPR 
calculate_fpr = function(T, F) {
  # No predictions need to be factor 1
  ctable <- table(T,F)
  N <- sum(ctable[1,])
  fp <- ctable[1,2]
  return (fp/N)
}

#Fit model using Naive Bayes Classifier

NB_fit = naiveBayes(y~.,data=train)

#Optimal tree is tree_dev, from 2c

#Predict using Naive Bayes fit. 
# Set type to raw to get probabilities
pred_Bayes = predict(NB_fit, newdata=test, type = "raw")

#pred with opt.tree
pred_tree = predict(tree_dev, newdata = test, type = "vector")


#DF to store results
bayes_tpr_fpr = data.frame("pi" = double(), "tpr" = double(), "fpr" = double())

tree_tpr_fpr = data.frame("pi" = double(), "tpr" = double(), "fpr" = double())

for (pi in seq(0.00, to=1, by=0.05)) {
  bayes = ifelse(pred_Bayes[,2] > pi, "yes", "no")
  tree = ifelse(pred_Bayes[,2] > pi, "yes", "no")
  
  bayes = as.factor(bayes)
  tree = as.factor(tree)
  
  if (any(levels(bayes) == "yes")) {
  } else {
    bayes <- factor(bayes, levels = c(levels(bayes), "yes"))
  }
  
  if (any(levels(bayes) == "no")) {
  } else {
    bayes <- factor(bayes, levels = c(levels(bayes), "no"))
    # reorder 
    bayes <- factor(bayes,levels = c("no", "yes"))
  }
  
  if (any(levels(tree) == "yes")) {
  } else {
    tree <- factor(tree, levels = c(levels(tree), "yes"))
  }
  
  if (any(levels(tree) == "no")) {
  } else {
    tree <- factor(tree, levels = c(levels(tree), "no"))
    #reorder so no is first. 
    tree <- factor(tree,levels = c("no", "yes"))
  }
  
  # Calculate TPR and FPR for bayes
  bayes_tpr_fpr <- rbind(bayes_tpr_fpr, 
  data.frame("pi"=pi, "tpr"=calculate_tpr(test$y, bayes), 
             "fpr"=calculate_fpr(test$y, bayes)))
  
  # Calculate TPR and FPR for tree.
  tree_tpr_fpr <- rbind(tree_tpr_fpr, 
  data.frame("pi"=pi, "tpr"=calculate_tpr(test$y,tree), 
             "fpr"=calculate_fpr(test$y, tree)))
  
}

bayes_tpr_fpr
tree_tpr_fpr
plot(tree_tpr_fpr$fpr, tree_tpr_fpr$tpr, col="orange", type="l",ylab="TPR", xlab="FPR", ylim=c(0,1), xlim=c(0,1))

plot(bayes_tpr_fpr$fpr, bayes_tpr_fpr$tpr, col="red", type="b",ylab="TPR", xlab="FPR", ylim=c(0,1), xlim=c(0,1))
points(tree_tpr_fpr$fpr, tree_tpr_fpr$tpr, type="l", col="blue")

#5 in an other way

# Q5 
## FOR THE OPTIMAL TREE ##
tree_model = pruned_tree
#prediction på testdatan
tree_pred = predict(tree_model, newdata=test, type="vector")
tree_TPR = c(rep(0,length(pi)))
tree_FPR = c(rep(0,length(pi)))
tree_p_yes = tree_pred[,2]
i=1
#loppar över pi
for (pi in seq(from=0.00, to=1, by=0.05)) {
  #sätter 
  tree_Y = ifelse(tree_p_yes > pi, 1, 0)
  confMa = table(test$y, tree_Y)
  # vår confMa bara en column, så vi lägger till noll,noll för att få en 4x4 matrix
  if (is.na(table(tree_Y)[2])) {
    if (colnames(confMa)[1] == "yes") {
      confMa = cbind(c(0,0), confMa)
    } else {
      confMa = cbind(confMa, c(0,0))
    }
  }
  #för varje pi sparar jag true positive rate
  tree_TPR[i] = confMa[2,2] / (confMa[2,1] + confMa[2,2])
  tree_FPR[i] = confMa[1,2] / (confMa[1,1] + confMa[1,2])
  i=i+1
}

## FOR THE NAIVE MODEL ##
# ny modell, prediction som tar fram sannolikheter iställer för vilken klass den tillhör
naive_model = naiveBayes(train$y~., train)
naive_pred = predict(naive_model, newdata=test, type="raw")
naive_TPR = c(rep(0,length(pi)))
naive_FPR = c(rep(0,length(pi)))
naive_p_yes = naive_pred[,2]
i=1
for (pi in seq(from=0.00, to=1, by=0.05)) {
  naive_Y = ifelse(naive_p_yes > pi, "yes", "no")
  confMa = table(test$y, naive_Y)
  if (is.na(table(naive_Y)[2])) {
    if (colnames(confMa)[1] == "yes") {
      confMa = cbind(c(0,0), confMa)
    } else {
      confMa = cbind(confMa, c(0,0))
    }
  }
  naive_TPR[i] = confMa[2,2] / (confMa[2,1] + confMa[2,2])
  naive_FPR[i] = confMa[1,2] / (confMa[1,1] + confMa[1,2])
  i=i+1
}

# Plot ROC-curves for both models
plot(tree_FPR, tree_TPR, type = "l", col="green", main="Green = tree, Blue = naive", xlab="FPR", ylab="TPR")
points(naive_FPR, naive_TPR, type = "l", col="blue")
points(tree.tpr.fpr$fpr, tree.tpr.fpr$tpr, type="l", col="blue")
points(tree_tpr_fpr$fpr, tree_tpr_fpr$tpr, type="l", col="red")
points(bayes_tpr_fpr$fpr, bayes_tpr_fpr$tpr, col="orange", type="b",ylab="TPR", xlab="FPR", ylim=c(0,1), xlim=c(0,1))
