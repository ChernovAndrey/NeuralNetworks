library(h5)
library(ggplot2)
library(car)
library(sp)

numLayers=17

name <- "p_val_shift"
name <- "p_val_variance"

list_p_val <-list()
file <- h5file("/home/andrey/datasetsNN/landScapes/landScape_3000_32/p_val_02_08_2018/p-value_global", 'r')
for (i in 0:16){
  p_val<-file[paste(name,toString(i),sep="")]
  print(p_val[])
  p_val<-sapply(p_val[],function(x){
    if (x <= 1e-16){
      return(1e-16)
    }else{
      return(x)
    }
  })
  
  print(p_val)
  list_p_val[[i+1]] <- (p_val[])
}
h5close(file)

boxplot(list_p_val,log="y",ylab =name,las=2,names = c("conv2d","max pool","bat_norm","drop_out","conv2d","max_pool",
                                                         "bat_norm","drop_out","conv2d","max_pool",
                                                         "bat_norm","reshape","conv1d","conv1d","conv1d","flatten","activation") )
abline(h=1e-4, col = "Red",log="y")

print(list_p_val[[5]])

boxplot(list_p_val)
abline(h=1e-4, col = "Red")

log(1e-4)
print(list_p_val[[10]])



log(0.1)


#merge
name <- "p_val_shift_n"
p_val_all=c()
for (i in 1:10){
  p_val<-file[paste(name,toString(i),"_0",sep="")]
  print(p_val[])
  print(p_val)
  p_val_all = c(p_val_all,p_val[])
}
print(p_val_all)

file["p_val_shift0"] <- p_val_all

file <- h5file("/home/andrey/datasetsNN/landScapes/landScape_3000_32/dissection/p-value")


