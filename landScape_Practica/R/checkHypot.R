library(sp)
library(h5)
library(ggplot2)
library(car)
library(abind)
file0 <- h5file("/home/andrey/datasetsNN/landScapes/landScape_3000_32/2/predict0.hdf5", 'r')
file1 <- h5file("/home/andrey/datasetsNN/landScapes/landScape_3000_32/2/predict1.hdf5", 'r')
res0 <- file0["predict0"]
res1 <- file1["predict1"]

wilcox.test(res0[],res1[],paired=FALSE)


file <- h5file("/home/andrey/datasetsNN/landScapes/landScape_3000_32/dissection/result1", 'r')
res0 <- file["res2_0"]
res1 <- file["res2_1"]
v0<-as.vector(res0[])
v1<- as.vector(res1[])
length(v0)
dim(res0)
dim(res1)
wilcox.test(v0,v1,paired=FALSE)


check_hypot_rand_samples <- function(num_repeat=100, num_samples = 5000){ 
  for (i in 1:num_repeat){
    res_layers0 = vector("list", length = 15)#15 - число слоев
    res_layers1 = vector("list", length = 15)
    for (j in 1:num_samples){
      rand_num = sample(1:60000, 1, replace = T)     # 60000- всего экземпляров
      nFile = (rand_num %/% (10000+1)) + 1       # 10000 - в одном файле
      file <- paste("/home/andrey/datasetsNN/landScapes/landScape_3000_32/dissection/result",toString(nFile),sep="")
      print(file)
      for(k in 0:14){ # по слоям
        res0 <-  file[paste("res",toString(k),"_","0",sep="")]
        res1 <- file[paste("res",toString(k),"_","1",sep="")]
        print(res0)
        res_layers0[[k+1]] = c(res_layers0[[k+1]],res0)
        res_layers1[[k+1]] = c(res_layers1[[k+1]],res1)
      }
    }
    print(res_layers0[[15]])
    #wilcox.test(res_layers0[[15]],res_layers1[[15]],paired=FALSE)
  }
}


check_hypot_rand_samples(1,100)
