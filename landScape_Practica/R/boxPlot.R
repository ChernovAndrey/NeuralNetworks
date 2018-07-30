library(h5)
library(ggplot2)
library(car)
library(sp)
file <- h5file("/home/andrey/datasetsNN/landScapes/landScape_3000_32/dissection/p-value", 'r')
numLayers=13
h5close(file)


name <- "p_val_shift"
name <- "p_val_variance"

list_p_val <-list()
for (i in 0:14){
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

boxplot(list_p_val,log="y",ylab ="shift",las=2,names = c("conv","max pool","bat_norm","conv","max_pool","bat_norm",
                                                    "conv","conv","drop_out","conv","max_pool","bat_norm","flatten",
                                                   "dense","dense") )
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


