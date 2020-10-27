#Set As Working Directory
setwd("~/STAT 515/Group1_Final Project")
library(GGally)
library(plotly)
library(plyr)
library(waffle)
library(ROCR)
library(randomForest)
library(ModelMetrics)
library(dplyr)
library(ggplot2)
library(ipred)
library(caret)
library(rpart)
library(rpart.plot)
library(aod)
# Read the csv file
m<-read.csv("student-mat.csv",header= TRUE,strip.white = TRUE,stringsAsFactors = T)
p<-read.csv("student-por.csv",header= TRUE,strip.white = TRUE,stringsAsFactors = T)

#merging to two files
students<-rbind(m,p)

#summarize the dataset
summary(students)
head(students,10)
glimpse(students)
#Data preprocessing
#Creating additional colunms  GG and Gavg for accuracy prediction and analysis after removing some columns.
students$GG<-cut(students$G3,c(-1,10,20))
students$Gavg<-(students$G1+students$G2+students$G3)/3
summary(students)
str(students)
studentsbc<-students

corp<-(cor(students[sapply(students, function(x) !is.factor(x))]))
corrplot::corrplot(corp,method="circle",order="hclust")

#Using Spearman correlation method analysis is made
ggcorr(corp, method = c("everything", "spearman"))+  ggtitle("Correlation Analysis")

students$Dalc<-as.factor(students$Dalc)
students$Dalc<-mapvalues(students$Dalc, from =1:5, to=c("Very Low","Low","Medium","High","Very High"))

students$Walc<-as.factor(students$Walc)
students$Walc<-mapvalues(students$Walc, from =1:5, to=c("Very Low","Low","Medium","High","Very High"))

#Colour combination using waffle
waffle.col <- c("#00d27f","#adff00","#f9d62e","#fc913a","#ff4e50")

#weekday alochol consumptions and grades
bp1<-ggplot(students, aes(x=Dalc, y=G1, fill=Dalc))+
  geom_boxplot()+
  theme_dark()+
  theme(legend.position="none")+
  scale_fill_manual(values=waffle.col)+
  xlab("Alcohol consumption")+
  ylab("Grade")+
  ggtitle("First period grade")
bp2<-ggplot(students, aes(x=Dalc, y=G2, fill=Dalc))+
  geom_boxplot()+
  theme_dark()+
  theme(legend.position="none")+
  scale_fill_manual(values=waffle.col)+
  xlab("Alcohol consumption")+
  ylab("Grade")+
  ggtitle("Second period grade")
bp3<-ggplot(students, aes(x=Dalc, y=G3, fill=Dalc))+
  geom_boxplot()+
  theme_dark()+
  theme(legend.position="none")+
  scale_fill_manual(values=waffle.col)+
  xlab("Alcohol consumption")+
  ylab("Grade")+
  ggtitle("Final period grade")
grid.arrange(bp1,bp2,bp3,ncol=3,top="Weekday Alcohol Consumption Analysis and Grades")

#weekend alcohol consumption and grades
bp4<-ggplot(students, aes(x=Walc, y=G1, fill=Walc))+
  geom_boxplot()+
  theme_dark()+
  theme(legend.position="none")+
  scale_fill_manual(values=waffle.col)+
  xlab("Alcohol consumption")+
  ylab("Grade")+
  ggtitle("First period grade")
bp5<-ggplot(students, aes(x=Walc, y=G2, fill=Walc))+
  geom_boxplot()+
  theme_dark()+
  theme(legend.position="none")+
  scale_fill_manual(values=waffle.col)+
  xlab("Alcohol consumption")+
  ylab("Grade")+
  ggtitle("Second period grade")
bp6<-ggplot(students, aes(x=Walc, y=G3, fill=Walc))+
  geom_boxplot()+
  theme_dark()+
  theme(legend.position="none")+
  scale_fill_manual(values=waffle.col)+
  xlab("Alcohol consumption")+
  ylab("Grade")+
  ggtitle("Final period grade")
grid.arrange(bp4,bp5,bp6,ncol=3,top="Weekend Alcohol Consumption Analysis and Grades")

#alcohol consumption by sex
p2 <- ggplot(data = students, aes(x = Walc, fill = sex))
p2 + geom_bar(stat = "count", position = "dodge") + 
  labs(title = "School Students Gender wise Weekend Alcohol Consumption by School", 
       x = "Weekend Alcohol Consuming Students(Rated(1-5))", 
       y = "Number of Students", col = "Gender") + theme_dark()

#PCA Analysis
#Principle Component Analysis
#Correlation on variables
ds<-studentsbc[,-c(31,32,33)]
ds
td<-data.frame(model.matrix(~.-1,data=ds)) 
cor_td <- cor(td, td, method = "spearman")
cor_df<- data.frame(cor=cor_td[1:40,41], varn = names(cor_td[1:40,41])) 
cor_df<- cor_df%>%mutate(cor_abs = abs(cor)) %>% arrange(desc(cor_abs))
plot(cor_df$cor_abs, type="l",main="Corelation graph of Processed Data", ylab="absolute cor-value")
summary(cor_df$cor_abs)

Abvvarn <- cor_df %>% filter(cor_abs>0.08)
fildf <- data.frame(td) %>% select(Gavg,one_of(as.character(Abvvarn$varn)))
head(fildf,10)
cp<-cor(fildf[sapply(fildf, function(x) !is.factor(x))])
corrplot::corrplot(cp,method="ellipse",order="hclust")

lm<-lm(data=fildf,Gavg~.)
summary(lm)

X<-fildf%>%select(-Gavg)
pca1= prcomp(X,scale. = TRUE,center= TRUE,)
plot(pca1,main="PCA on highly correlated variables")
#summary of PCA Fit
summary(pca1)

dfpca1<-data.frame(pca1$x)
dfpca1<-dfpca1[,-c(11,12,13,14,15,16,17,18)]
dfpca1$Gavg=fildf$Gavg

#fitting PCA model
mod<-lm(data=dfpca1,Gavg~.)
summary(mod)

PCt <- function(PC, x="PC1", y="PC2") {
  data <- data.frame( PC$x)
  plot <- ggplot(data, aes_string(x=x, y=y))
  datapc <- data.frame(varnames=row.names(PC$rotation), PC$rotation)
  mult <- min(
    (max(data[,y]) - min(data[,y])/(max(datapc[,y])-min(datapc[,y]))),
    (max(data[,x]) - min(data[,x])/(max(datapc[,x])-min(datapc[,x])))
  )
  datapc <- transform(datapc,
                      v1 = .7 * mult * (get(x)),
                      v2 = .7 * mult * (get(y))
  )
  plot <- plot + coord_equal() + geom_text(data=datapc, aes(x=v1, y=v2, label=varnames), size = 3, vjust=1, color="blue")
  plot <- plot + geom_segment(data=datapc, aes(x=0, y=0, xend=v1, yend=v2), arrow=arrow(length=unit(0.2,"cm")), alpha=0.5, color="black")
  plot+ggtitle("Top principle components variable wise")+theme_light()
}
#Principle Component Analysis Fit
PCt(pca1)

#linear regression
lr1<-glm(students$G3~students$G1+students$G2,data=students)
summary(lr1)
sigma(lr1)*10/mean(students$G3)
lr2<-glm(students$Gavg~students$sex+students$Medu+students$Fedu+students$absences+students$failures+students$Dalc+students$Walc,data=students)
summary(lr2)
sigma(lr2)*10/mean(students$Gavg)
confint.lm(lr2)
wald.test(b = coef(lr2), Sigma = vcov(lr2), Terms = 4:6)

#LOGISTIC REGRESSION
stud<-subset(students,select=c(age,Medu,Fedu,traveltime,studytime,failures,freetime,goout,Dalc,Walc,health,absences,G1,G2,G3,GG))
set.seed(12345)
row<-nrow(stud)
trainindex<-sample(row,0.60*row,replace = FALSE)
training<-stud[trainindex,]
validation<-stud[-trainindex,]
mylogit4<-glm(G3~age+traveltime+studytime+failures+freetime+goout+Dalc+Walc+health+G1+G2,data=training,family=binomial)
summary(mylogit4)
str(stud$GG)
mylogit.step = step(mylogit4, direction='backward')
gfit<-glm(GG ~ studytime + failures + Walc + G1 + G2,data=training,family=binomial)
summary(gfit)
mylogit.probs2<-predict(mylogit4,validation,type="response")
mylogit.pred2 = rep("BELOW AVERAGE", 0.4*row)
mylogit.pred2[mylogit.probs2 >0.5] = "good grades"
table(mylogit.pred2, validation$GG)
#Accuracy: 93.77%



#good grades fit
mylogit.probs1<-predict(gfit,validation,type="response")
mylogit.pred2 = rep("BELOW AVERAGE", 0.4*row)
mylogit.pred2[mylogit.probs1 >0.5] = "good grades"
table(mylogit.pred2, validation$GG)
#Accuracy: 93.77%


#decision tree
fitr2<-rpart(GG~age+Medu+Fedu+traveltime+studytime+failures+freetime+goout+Dalc+Walc+health+G1+G2,data=training,method="class")
plot<-rpart.plot(fitr2,type=2,extra="auto",main="Decision Tree")
pred<-predict(fitr2,validation,type="class")
Tr2<-confusionMatrix(validation$GG,pred)
Tr2
Tf<-table(validation$GG,pred)
Tf
#accuracy 93.5%


#Random Forest
fit<-randomForest(G3~age+Medu+Fedu+traveltime+studytime+failures+freetime+goout+Dalc+Walc+health+G1+G2,data=training,ntree=500,type="class",mtry=3)
plot<-varImpPlot(fit)
imp<-varImp
imp<-as.data.frame(imp)
pf<-predict(fit,validation,type="class")
pf[1:5]
tt<-table(validation$G3,pf)
tt
cm
fit2<-randomForest(GG~age+Medu+Fedu+traveltime+studytime+failures+freetime+goout+Dalc+Walc+health+G1+G2,data=training,ntree=500,type="class",mtry=3)
plot2<-varImpPlot(fit2,type=2,color="orange")

predictrand<-predict(fit2,validation,type="class")
T1<-table(validation$GG,predictrand)
T1
#Accuracy 93.77%

#removing important variables g1,g2 and g3
fitr1<-randomForest(GG~.-G2-G1-G3,data=training,ntree=500,type="class",mtry=3)
plot2<-varImpPlot(fitr1,type=2)
pre<-predict(fitr1,validation,type="class")
Tr1<-table(validation$GG,pre)
Tr1
#We see a low accuracy here.
predictrand[1:5]
#bagging
BTree= bagging(GG~age+Medu+Fedu+traveltime+studytime+failures+freetime+goout+Dalc+Walc+health+G1+G2,data=training)
B_yHat = predict(BTree,validation)
BPR = postResample(pred=B_yHat, obs=validation$GG)
BPR
#accuracy 92.34%


