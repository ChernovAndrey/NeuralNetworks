library(h5)
library(ggplot2)
library(car)
library(sp)
file <- h5file("/home/andrey/datasetsNN/landScapes/landScape_3000_32/dissection/p-value", 'r')
numLayers=13
x <- file[paste("x",toString(numLayers),"_0",sep="")]
y0 <- file[paste("y",toString(numLayers),"_0",sep="")]
y1 <- file[paste("y",toString(numLayers),"_1",sep="")]
h5close(file)
len <- dim(x)
start = 2

p_val_shift <- file["p_val_shift10"]

boxplot(p_val_shift[])
plot(p_val_shift[])
sapply(p_val_shift[],function(x){
  if (x == 0){
    x = 1e-16
  }
})

name <- "p_val_variance"
list_p_val <-list()
for (i in 3:14){
  p_val<-file[paste(name,toString(i),sep="")]
  print(p_val[])
  p_val<-sapply(p_val[],function(x){
    if (x == 0){
      return(1e-16)
    }else{
      return(x)
    }
  })
  print(p_val)
  list_p_val[[i-2]] <- p_val[]
}

boxplot(list_p_val,log="y",ylim(0,1))
abline(h=log(1e-4), col = "Red")
print(list_p_val[[1]])

log(1e-4)        
log10(1e-4)

boxplot(list_p_val)
abline(h=1e-4, col = "Red")
