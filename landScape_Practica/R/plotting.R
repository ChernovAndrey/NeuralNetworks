library(sp)
library(h5)
library(ggplot2)
library(car)
file <- h5file("/home/andrey/datasetsNN/landScapes/landScape_3000_32/dissection/allOutLayers.hdf5", 'r')
numLayers=13
x <- file[paste("x",toString(numLayers),"_0",sep="")]
y0 <- file[paste("y",toString(numLayers),"_0",sep="")]
y1 <- file[paste("y",toString(numLayers),"_1",sep="")]
h5close(file)
len <- dim(x)
start = 2
df <- data.frame(x[start:len],y0[start:len],y1[start:len])
ggplot(df, aes(x[start:len])) +                    # basic graphical object
  geom_line(aes(y=y0[start:len]), colour="red") +  # first layer
  geom_line(aes(y=y1[start:len]), colour="green")  # second layer

dim(y0[])
plot(x[start:len],y0[start:len],type="l",col="red")
par(new=TRUE)
plot(x[start:len],y1[start:len],type="l",col="green")



x[10]
y1[1:10]
len


sum(y0[])
sum(y1[])

poisX = 1:100
lambda = 20
poisY = ( ( lambda^poisX )/factorial(poisX))*exp(-lambda)
plot(poisX,poisY,type="l")





#wilcox.test(y0[],y1[],paired=TRUE)
wilcox.test(y0[],y1[],paired=FALSE)


t1 =c(0,2,3,10)
t2 =c(1,5,4,-11)
wilcox.test(t1,t2,paired=FALSE)


t1 =c(-1,2,3,10)
t2 =c(1,-2,4,-11)
wilcox.test(t1,t2,paired=TRUE)




