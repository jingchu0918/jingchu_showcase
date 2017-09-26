####========== Chinese University of Hong Kong ===========####
####======= RMSC4102 (16-17 sem2) Group Project ==========####
#                                                            #
#   Modeling Default Risk wish Support Vector Machines       #
#                                                            #
####======================================================####
##=====Group members=========##                   
#                             #
# CHU   Jing      1155047032  #      
# XIE   Peijie    1155047076  #    
# ZHANG Mengqi    1155047022  #          
# ZHOU  Zehui     1155046910  #     
##===========================##


##===========================##
##                           ##
## Install and load packages ##
##                           ##
##===========================##

if(!require('e1071')) install.packages('e1071')
if(!require('MLmetrics')) install.packages('MLmetrics')
if(!require('RColorBrewer')) install.packages('RColorBrewer')
if(!require('rgl')) install.packages('rgl')
if(!require('misc3d')) install.packages('misc3d')
if(!require('car')) install.packages('car')
if(!require('ggplot2')) install.packages('ggplot2')
if(!require('reshape2')) install.packages('reshape2')

library(e1071)
library(MLmetrics)
library(RColorBrewer)
library(rgl)
library(misc3d)
library(car)
library(ggplot2)
library(reshape2)


##===========================##
##                           ##
##         Part 1            ##
##        Functions          ##
##                           ##
##===========================##

##Calculate Accuracy Ratio (AR)##
accuracy.ratio <- function(pr,y){
  y<-as.numeric(y)
  ysort<-y[order(pr,decreasing=T)]
  perc<-cumsum(ysort)/sum(ysort)
  pop<-(1:length(ysort))/length(ysort)
  perfect<-cumsum(sort(ysort,decreasing = TRUE))/sum(ysort)
  return((Area_Under_Curve(pop,perc)-0.5)/(Area_Under_Curve(pop,perfect)-0.5))
}

###Forward stepwise predictor selection for SVM##
##================================Remark 1 [About sampling method]================================##
# 1. Size of training dataset : Size of testing dataset = 26.9 : 1                                 #
# 2. Bookstrap sampling is used to obtain 30 samples for training and testing dataset respectively #
# 3. Each sample has equal class size ('0' = solvencies / '1' = insolvencies)                                                        #
##=======================Remark 2 [About forward stepwise predictor selection]====================##
# 1. The function add one varied predictor to the fixed predictors as the input for SVM            #
# 2. svm.AR is a matrix with number of rows = number of varied predictors                          #
#                            number of cols = 30 (sample 30 times)                                 #
# 3. The return value svm.AR.median is a column vector with size = number of varied predictors     #
##================================================================================================##

svm.stepwise<-function(fixed,varied,samples=30,C=1,r=1){
  fixed.pred.name=colnames(fixed)
  
  set.seed(1234)
  d0<-d[d$class==0,]
  d1<-d[d$class==1,]
  ID0.training.sample<-sample(nrow(d0),6322)
  ID1.training.sample<-sample(nrow(d1),269)
  d0.training.sample<-d0[ID0.training.sample,]
  d0.testing.sample<-d0[-ID0.training.sample,]
  d1.training.sample<-d1[ID1.training.sample,]
  d1.testing.sample<-d1[-ID1.training.sample,]
  
  nv<-ncol(varied)
  svm.AR<-matrix(rep(NA,samples*nv),nrow = nv)
  
  for (subsample in 1:samples){
    ID0.training.subsample<-sample(nrow(d0.training.sample),269)
    ID0.testing.subsample<-sample(nrow(d0.testing.sample),100)
    d0.training.subsample<-d0.training.sample[ID0.training.subsample,]
    d0.testing.subsample<-d0.testing.sample[ID0.testing.subsample,]
    training<-rbind(d0.training.subsample,d1.training.sample)
    testing<-rbind(d0.testing.subsample,d1.testing.sample)
    
    for (pred.index in 1:nv){
      varied.pred.name<-colnames(varied)[pred.index]
      fixed<-as.matrix(fixed)
      if (ncol(fixed)==0){
        temp.data<-training[,c(varied.pred.name,"class")]
      }else{
        temp.data<-training[,c(fixed.pred.name,varied.pred.name,"class")]
      }
      
      svm.training.model<-svm(class ~ .,data=as.data.frame(temp.data), cost=C,gamma=r,
                              type = "C-classification",kernal = "radial",probability=T,decision.value=T)
      
      if (ncol(fixed)==0){
        temp.test.data<-as.data.frame(testing[,c(varied.pred.name)])
        colnames(temp.test.data)<-varied.pred.name
      }else{
        temp.test.data<-testing[,c(fixed.pred.name,varied.pred.name)]
      }
      
      svm.testing.value<-predict(svm.training.model,as.data.frame(temp.test.data),decision.values=T,probability = T)
      svm.AR[pred.index,subsample]<-accuracy.ratio(as.vector(attr(svm.testing.value,"probabilities")[,2]),as.vector(testing$class))

    }
  }
  svm.AR.median<-apply(svm.AR,1,median)
  names(svm.AR.median)<-names(varied)
  svm.AR.median<-as.matrix(svm.AR.median)

  return(svm.AR.median)
}

##Adjusting decision threshold##
svm.decision<-function(probability,cost1,cost2){
  decision<-vector()
  for (i in 1:length(probability)){
    if (probability[i]>cost1/(cost1+cost2)){
      decision[i]<-1
    }else{
      decision[i]<-0
    }
  }
  return(decision)
}

##===========================##
##                           ##
##         Part 2            ##
##      Data Cleaning        ##
##                           ##
##===========================##

##Read in dataset###
d<-read.csv("data.csv")
d<-d[complete.cases(d),]
row.names(d)<-c(1:nrow(d))

##Modify extreme values##
for (j in 1 : (ncol(d)-1)) {
  d[,j]<-as.numeric(d[,j]) 
  low=quantile(d[,j],0.05)
  up=quantile(d[,j],0.95)
  for (i in 1:nrow(d)){
    if (d[i,j]>up) d[i,j]=up
    if (d[i,j]<low) d[i,j]=low 
  }}


##Data standardization##
std <- function(x) {
  xn = nrow(x)
  mat1 = rep(1,xn)
  xminmat = mat1%*%t(as.numeric(apply(x,2,min)))
  xmaxmat = mat1%*%t(as.numeric(apply(x,2,max)))
  xstand = (x-xminmat)/(xmaxmat-xminmat)
  xstand
}
d<-std(d)

##Correlation of predictors##
z <- cor(d[,-24])     # obtain the correlation matrix
z.m <- melt(z)        # Cartesian product
cormatrix <- ggplot(z.m, aes(Var1, Var2, fill = value)) +
  geom_tile() + scale_fill_gradient2(low = "lightblue", mid = "white", high = "coral") +
  labs(x='Variables',y='Variables',title = 'Correlation heatmap')
cormatrix <- cormatrix + theme(
  rect = element_rect(fill = "transparent") #background of the panel
)
cormatrix
ggsave(cormatrix, filename = "corrlation heatmap.png",  bg = "transparent")

##Drop variables that have similar financial meanings and are highly-correlated with selected predictors######
d<-d[,-c(1,3,4,5,7,8,10,14,16)]

##3D plot of the dataset indicates that the data is linear non-seperable##
d0.plot<-d[d$class==0,]
d1.plot<-d[d$class==1,]
d.plot<-rbind(d0.plot[1:369,],d1.plot[1:369,])
scatter3d(d.plot$x2,d.plot$x11,d.plot$x13,groups=as.factor(d.plot$class),
          surface=F, grid = F, radials = 5, 
          surface.col = c("dodgerblue4","coral3"),
          axis.col = c("black","black","black") )


##===========================##
##                           ##
##         Part 3            ##
##        SVM Model          ##
##                           ##
##===========================##

##Select the first predictor using default C and r##
AR.1.default<-svm.stepwise(data.frame(NULL),data.frame(d[,-15]))
first.pred.defaut<-rownames(AR.1.default)[which.max(AR.1.default)]
first.pred.defaut 
##The first predictor selected is x15: Quick Ratio

##Find the optimal C and r using x15 as input##
set.seed(4102)
d0<-d[d$class==0,]
d1<-d[d$class==1,]
tune.data<-rbind(d1,d0[sample(nrow(d0),369),])

set.seed(4102)
svm.para<-tune.svm(tune.data[,7],tune.data[,15],
                   gamma = c(0.1*2^(-3:0),seq(0.15,0.4,0.05)), cost = c(0.5*2^(-1:4),12,16))
svm.para 
##The optimal value is C=8, r=0.05

##Plot the sensitivity of error rates to the value of C and r
par(bg=NA) 
plot(svm.para,type = "contour",
     color.palette = hsv_palette(h = 0.05, from = 0.65, to = 0, v = 1),
     nlevels = 12,main = "")
par(bg=NA) 
plot(svm.para,type = "contour",
     color.palette = hsv_palette(h = 0.05, from = 0.65, to = 0, v = 1),
     transform.x = log2, transform.y = log2,
     xlab = "log2 (gamma)", ylab = "log2 (cost)",
     nlevels = 12,main = "")
par(bg=NA) 
plot(svm.para,type = "perspective",main = "")
par(bg=NA) 
plot(svm.para,type = "perspective",
     transform.x = log2, transform.y = log2,
     xlab = "log2 (gamma)", ylab = "log2 (cost)",
     nlevels = 12,main = "")

##SVM with one input predictor and optimal C and r##
AR.1<-svm.stepwise(data.frame(NULL),d[,-15],C=8,r=0.05)
pred.1<-rownames(AR.1)[which.max(AR.1)]
pred.1 
AR.1
##The first predictor selected is x15: Quick Ratio (AR = 0.4373)

##Selection of the second predictor##
AR.2<-svm.stepwise(subset(d,select=x15),d[,-c(7,15)],C=8,r=0.05)
pred.2<-rownames(AR.2)[which.max(AR.2)]
pred.2 
AR.2
##The second predictor selected is x2: Net Profit Margin (AR = 0.5198)

##Selection of the third predictor##
AR.3<-svm.stepwise(d[,c(1,7)],d[,-c(1,7,15)],C=8,r=0.05)
pred.3<-rownames(AR.3)[which.max(AR.3)]
pred.3 
AR.3
##The third predictor selected is x6: EBITDA (AR = 0.5542)

##Selection of the fourth predictor##
AR.4<-svm.stepwise(d[,c(1,2,7)],d[,-c(1,2,7,15)],C=8,r=0.05)
pred.4<-rownames(AR.4)[which.max(AR.4)]
pred.4 
AR.4
##The fourth predictor selected is x22: Account Payable Turnover (AR = 0.5614)

##Selection of the fifth predictor##
AR.5<-svm.stepwise(d[,c(1,2,7,13)],d[,-c(1,2,7,13,15)],C=8,r=0.05)
pred.5<-rownames(AR.5)[which.max(AR.5)]
pred.5 
AR.5
##The fifth predictor selected is x23: Log (Total Asset)  (AR = 0.5872)

##Selection of the sixth predictor####
AR.6<-svm.stepwise(d[,c(1,2,7,13,14)],d[,-c(1,2,7,13,14,15)],C=8,r=0.05)
pred.6<-rownames(AR.6)[which.max(AR.6)]
pred.6 
AR.6
#The sixth predictor selected is x20: Inventory Turnover (AR = 0.6026)

##Selection of the seventh predictor####
AR.7<-svm.stepwise(d[,c(1,2,7,11,13,14)],d[,-c(1,2,7,11,13,14,15)],C=8,r=0.05)
pred.7<-rownames(AR.7)[which.max(AR.7)]
pred.7 #the 7th predictor selected (x11 0.5879588)
AR.7
#The highest AR = 0.5880 < 0.6026
#AR starts to decline
#Forward stepwise selection stops


##===========================##
##                           ##
##         Part 4            ##
##    Model Visualiztion     ##
##                           ##
##===========================##

##2-D plot of SVM##
set.seed(4102)
d0<-d[d$class==0,]
d1<-d[d$class==1,]
ID0.training.sample<-sample(nrow(d0),6322)
ID1.training.sample<-sample(nrow(d1),269)
d0.training.sample<-d0[ID0.training.sample,]
d0.testing.sample<-d0[-ID0.training.sample,]
d1.training.sample<-d1[ID1.training.sample,]
d1.testing.sample<-d1[-ID1.training.sample,]
training<-rbind(d0.training.sample[sample(nrow(d0.training.sample),269),],d1.training.sample)
testing<-rbind(d0.testing.sample[sample(nrow(d0.testing.sample),100),],d1.testing.sample)
svm.training.model<-svm(class ~ .,data=training[,c(1,7,15)], cost=8,gamma=0.05,
                        type = "C-classification",kernal = "radial",probability=T,decision.value=T)
plot(svm.training.model,training[,c(1,7,15)],col = brewer.pal(3,"Blues"))

##3-D plot of SVM##
n<-100
nnew<-50
plot.data<-training[,c(1,2,7,15)]
fit = svm(class ~ ., data=plot.data)
color=c("coral","cornflowerblue")
plot3d(plot.data[,-4], col=color[plot.data$class+1],size=5)
newdat.list<-lapply(plot.data[,-4], function(x) seq(min(x), max(x), len=nnew))
newdat<-expand.grid(newdat.list)
newdat.pred<-predict(fit, newdata=newdat, decision.values=T)
newdat.dv<-attr(newdat.pred, 'decision.values')
newdat.dv<-array(newdat.dv, dim=rep(nnew, 3))
contour3d(newdat.dv, color="white", level=0, alpha=0.5,
          x=newdat.list$x2, y=newdat.list$x6, z=newdat.list$x15, add=T)


##===========================##
##                           ##
##         Part 5            ##
##    Practical Concerns     ##
##                           ##
##===========================##

##Error table of SVM where solvencies : insolvencies = 1 : 1
set.seed(4102)
d0<-d[d$class==0,]
d1<-d[d$class==1,]
ID0.training.sample<-sample(nrow(d0),6322)
ID1.training.sample<-sample(nrow(d1),269)
d0.training.sample<-d0[ID0.training.sample,]
d0.testing.sample<-d0[-ID0.training.sample,]
d1.training.sample<-d1[ID1.training.sample,]
d1.testing.sample<-d1[-ID1.training.sample,]
training<-rbind(d0.training.sample[sample(nrow(d0.training.sample),269),],d1.training.sample)
testing<-rbind(d0.testing.sample[sample(nrow(d0.testing.sample),100),],d1.testing.sample)
svm.training.model<-svm(class ~ .,data=training[,c(1,2,7,11,13,14,15)], cost=8,gamma=0.05,
                        type = "C-classification",kernal = "radial",probability=T,decision.value=T)
svm.testing.predict<-predict(svm.training.model,testing[,c(1,2,7,11,13,14)])
ER.table<-table(svm.testing.predict,testing$class)
ER.table

##Error table of SVM where solvencies : insolvencies = 23.5 : 1
set.seed(4102)
d0<-d[d$class==0,]
d1<-d[d$class==1,]
ID0.training.sample<-sample(nrow(d0),6322)
ID1.training.sample<-sample(nrow(d1),269)
d0.training.sample<-d0[ID0.training.sample,]
d0.testing.sample<-d0[-ID0.training.sample,]
d1.training.sample<-d1[ID1.training.sample,]
d1.testing.sample<-d1[-ID1.training.sample,]
training<-rbind(d0.training.sample,d1.training.sample)
testing<-rbind(d0.testing.sample,d1.testing.sample)
svm.training.model<-svm(class ~ .,data=training[,c(1,2,7,11,13,14,15)], cost=8,gamma=0.05,
                        type = "C-classification",kernal = "radial",probability=T,decision.value=T)
svm.testing.predict<-predict(svm.training.model,testing[,c(1,2,7,11,13,14)])
ER.table<-table(svm.testing.predict,testing$class)
ER.table
##The error table shows that no defaults are predicted correctly

##=================================##
##                                 ##
##             Part 6              ##
##    Incorporate Class-dependent  ##
##      Misclassification Costs    ##
##                                 ##
##=================================##


##Error table of SVM with C=80000 (solvencies : insolvencies = 23.5 : 1)
svm.training.model<-svm(class ~ .,data=training[,c(1,2,7,11,13,14,15)], cost=80000,gamma=0.05,
                        type = "C-classification",kernal = "radial",probability=T,decision.value=T)
svm.testing.predict<-predict(svm.training.model,testing[,c(1,2,7,11,13,14)])

ER.table<-table(svm.testing.predict,testing$class)
ER.table


##Weighted SVM

##Function for forward stepwise selection as before expect for adjusting the weight in SVM 
svm.weight.stepwise<-function(fixed,varied,weight0=1,weight1=1,samples=1,C=1,r=1){
  fixed.pred.name=colnames(fixed)
  
  set.seed(4102)
  d0<-d[d$class==0,]
  d1<-d[d$class==1,]
  ID1.testing.sample<-as.numeric(row.names(d1.testing.sample))
  
  nv<-ncol(varied)
  svm.AR<-matrix(rep(NA,samples*nv),nrow = nv)
  svm.ER<-matrix(rep(NA,samples*nv),nrow = nv)
  
  for (subsample in 1:samples){
    ID0.training.sample<-sample(nrow(d0),6322)
    ID1.training.sample<-sample(nrow(d1),269)
    d0.training.sample<-d0[ID0.training.sample,]
    d0.testing.sample<-d0[-ID0.training.sample,]
    d1.training.sample<-d1[ID1.training.sample,]
    d1.testing.sample<-d1[-ID1.training.sample,]
    training<-rbind(d0.training.sample,d1.training.sample)
    testing<-rbind(d0.testing.sample,d1.testing.sample)
    
    for (pred.index in 1:nv){
      varied.pred.name<-colnames(varied)[pred.index]
      fixed<-as.matrix(fixed)
      if (ncol(fixed)==0){
        temp.data<-training[,c(varied.pred.name,"class")]
      }else{
        temp.data<-training[,c(fixed.pred.name,varied.pred.name,"class")]
      }
      
      svm.training.model<-svm(class ~ .,data=as.data.frame(temp.data), cost=C,gamma=r,
                              class.weights=c("0"=weight0, "1"=weight1),
                              type = "C-classification",kernal = "radial",probability=T,decision.value=T)
      
      if (ncol(fixed)==0){
        temp.test.data<-as.data.frame(testing[,c(varied.pred.name)])
        colnames(temp.test.data)<-varied.pred.name
      }else{
        temp.test.data<-testing[,c(fixed.pred.name,varied.pred.name)]
      }
      
      svm.testing.value<-predict(svm.training.model,as.data.frame(temp.test.data),decision.values=T,probability = T)
      svm.AR[pred.index,subsample]<-accuracy.ratio(as.vector(attr(svm.testing.value,"probabilities")[,2]),as.vector(testing$class))
      
      #ER.table<-table(svm.testing.predict,testing$class)
      #svm.ER[pred.index,subsample]<-(ER.table[1,2]+ER.table[2,1])/sum(ER.table)
    }
  }
  svm.AR.median<-apply(svm.AR,1,median)
  names(svm.AR.median)<-names(varied)
  svm.AR.median<-as.matrix(svm.AR.median)
  
  #svm.ER.median<-apply(svm.ER,1,median)
  #names(svm.ER.median)<-names(varied)
  #svm.ER.median<-as.matrix(svm.ER.median)
  
  return(svm.AR.median)
  #return(svm.ER.median)
  #svm.AR
}


##Error table of weighted SVM (solvencies : insolvencies = 23.5 : 1)#####
##Remark
##1. Our objection is to make the expected cost as small as possible
##2. We adjust the weight manually to reach the objection
##3. The optimal weight is '0'/'1' = 1 : 46
set.seed(4102)
d0<-d[d$class==0,]
d1<-d[d$class==1,]
ID0.training.sample<-sample(nrow(d0),6322)
ID1.training.sample<-sample(nrow(d1),269)
d0.training.sample<-d0[ID0.training.sample,]
d0.testing.sample<-d0[-ID0.training.sample,]
d1.training.sample<-d1[ID1.training.sample,]
d1.testing.sample<-d1[-ID1.training.sample,]
training<-rbind(d0.training.sample,d1.training.sample)
testing<-rbind(d0.testing.sample,d1.testing.sample)
svm.training.model<-svm(class ~ .,data=training[,c(7,1,2,13,14,11,15)], cost=80000,gamma=0.05,
                        class.weights=c("0"=1, "1"=46),
                        type = "C-classification",kernal = "radial",probability=T,decision.value=T)
svm.testing.predict<-predict(svm.training.model,testing[,c(7,1,2,13,14,11)])
ER.table<-table(svm.testing.predict,testing$class)
ER.table
ER.table[2,1]+ER.table[1,2]*35   #Objection function: expected cost
##contour plot of weighted SVM
par(bg=NA)
t<-cbind(testing[,c(7,1)],attr(svm.testing.value,"probabilities")[,2]*10)
colnames(t)<-c("x15","x2","probabilities")
require(akima)
resolution <- 0.01 # you can increase the resolution by decreasing this number (warning: the resulting dataframe size increase very quickly)
a <- interp(x=t$x15, y=t$x2, z=t$"probabilities", xo=seq(min(t$x15),max(t$x15),by=resolution),
            yo=seq(min(t$x2),max(t$x2),by=resolution), duplicate="mean")
filled.contour(a, zlim = range(t$"probabilities", finite = TRUE),color.palette=terrain.colors) 
color=c("skyblue4","tomato3")
points(x=(t$x15-0.08)/1.09,y=t$x2,col=color[testing$class+1],pch=testing$class+16,cex=testing$class/2+0.4)



##Adjusting decision threshold##
svm.training.model<-svm(class ~ .,data=training[,c(7,1,2,13,14,11,15)], cost=8,gamma=0.05,
                        type = "C-classification",kernal = "radial",probability=T,decision.value=T)
svm.testing.predict<-predict(svm.training.model,testing[,c(7,1,2,13,14,11)])

##Manual adjustion of threshold to make the objection function minimum
svm.threshold<-function(threshold,svm.testing.predict){
  
  #ER.table<-table(svm.testing.predict,testing$class)
  ER.table<-table(svm.testing.predict,testing$class)
  ER.table[2,1]+ER.table[1,2]*35
}

#####Optimal decision thredshold = 1/27#####
set.seed(4102)
d0<-d[d$class==0,]
d1<-d[d$class==1,]
ID0.training.sample<-sample(nrow(d0),6322)
ID1.training.sample<-sample(nrow(d1),269)
d0.training.sample<-d0[ID0.training.sample,]
d0.testing.sample<-d0[-ID0.training.sample,]
d1.training.sample<-d1[ID1.training.sample,]
d1.testing.sample<-d1[-ID1.training.sample,]
training<-rbind(d0.training.sample,d1.training.sample)
testing<-rbind(d0.testing.sample,d1.testing.sample)
svm.training.model<-svm(class ~ .,data=training[,c(1,2,7,13,14,15)], cost=8,gamma=0.1,
                        type = "C-classification",kernal = "radial",probability=T,decision.value=T)
svm.testing.value<-predict(svm.training.model,testing,probability=T,decision.values=T)
probability<-as.vector(attr(svm.testing.value,"probabilities")[,2])
modify.pred<-svm.decision(probability,1,35)
table(modify.pred,testing$class)


###############################End of R Code##############################################
