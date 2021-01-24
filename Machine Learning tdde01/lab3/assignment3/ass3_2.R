library(neuralnet)
set.seed(1234567890)
Var <- runif(500, 0, 10)
mydata <- data.frame(Var, Sin=sin(Var))
tr <- mydata[1:25,] # Training
te <- mydata[26:500,] # Test


winit <- runif(25,-1,1)# weight initilization. 
tr<-as.matrix(tr)
set.seed(1234567890)
nn <- neuralnet(Sin ~ Var, data=tr, hidden=c(17,20,25), 
                startweights=winit) 

plot(tr, cex=2, main="Sin~Var")
points(te, col = "blue", cex=1)
points(te[,1],predict(nn,te), col="red", cex=1)
sum((te[,1] - predict(nn, te))^2/450)
##########################################################

Var <- runif(500, 0, 20)
newdata <- data.frame(Var, Sin=sin(Var))

plot(newdata, col="blue", main="new Data 0:20", ylim=c(-2,1))
points(tr, cex=2)
points(newdata[,1], predict(nn, newdata), col="red", cex=1)


###########################################################


Var <- runif(500, 0, 10)
newdata2 <- data.frame(Var, Sin=sin(Var))
nn_2 <- neuralnet(Var ~ Sin, data=newdata2, hidden=c(5), 
                  startweights=winit) 

plot(newdata2$Sin, newdata2$Var, main="Var~Sin", col="blue")
points(newdata2[,2] , predict(nn_2, newdata2), col="red", cex=1)