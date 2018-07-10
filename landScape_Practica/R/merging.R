file <- h5file("/home/andrey/datasetsNN/landScapes/landScape_3000_32/dissection/outLayers", 'r')
file2 <- h5file("/home/andrey/datasetsNN/landScapes/landScape_3000_32/dissection/outLayers2", 'r')
numLayers=14
x <- file[paste("x",toString(numLayers),"_0",sep="")]
y0 <- file[paste("y",toString(numLayers),"_0",sep="")]
y1 <- file[paste("y",toString(numLayers),"_1",sep="")]

x2 <- file2[paste("x",toString(numLayers),"_0",sep="")]
y20 <- file2[paste("y",toString(numLayers),"_0",sep="")]
y21 <- file2[paste("y",toString(numLayers),"_1",sep="")]
h5close(file)

allFile <- h5file("allOutLayers.hdf5")
for (i in 0:14) { # 14- кол-во слоев
  file <- h5file("/home/andrey/datasetsNN/landScapes/landScape_3000_32/dissection/outLayers", 'r')
  x <- file[paste("x",toString(i),"_0",sep="")]
  y0 <- file[paste("y",toString(i),"_0",sep="")]
  y1 <- file[paste("y",toString(i),"_1",sep="")]
  len = dim(x)
  allX <-x[1:len]
  allY0 <- y0[1:len]
  allY1 <- y1[1:len]
  for (j in 2:6){ # 4- кол-во файлов
    file <- h5file(paste("/home/andrey/datasetsNN/landScapes/landScape_3000_32/dissection/outLayers",toString(j),sep=""), 'r')
    x <- file[paste("x",toString(i),"_0",sep="")]
    y0 <- file[paste("y",toString(i),"_0",sep="")]
    y1 <- file[paste("y",toString(i),"_1",sep="")]
    
    allY0  = allY0 + y0[1:len]
    allY1  = allY1 + y1[1:len]
    h5close(file)
    
  }
  allFile[paste("x",toString(i),"_0",sep="")] <- allX
  allFile[paste("y",toString(i),"_0",sep="")] <- allY0
  allFile[paste("y",toString(i),"_1",sep="")] <- allY1
  
}
h5close(allFile)
sum(allY0)
sum(allY1)







