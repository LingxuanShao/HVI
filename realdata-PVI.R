library("foreach")
library("doParallel")
require("ggplot2")
library("expm")
library("mvtnorm")
library("SuperLearner")
library("randomForest")
library("glmnet")
library("nnet")

#kernel
Ker=function(s,h){
  temp=ifelse((s-h)>0,0,ifelse((s+h)<0,0,((1-(abs(s/h))**3)**3)*70/(h*81)))
  return(temp);
}


SL.glmnet.lasso <- function(...) {
    SL.glmnet(..., alpha = 1)
}


# data
M=5;
tau=0.05;
B=1000;
dat <- readRDS("~/Documents/Project 10 HVI with DaiGuorong/dat.RDS");
set.seed(79)
dat=dat[1:45500,]
N=dim(dat)[1]
age_mu=mean(dat[,3])
age_sd=sd(dat[,3])
# age_mu=47.87608; age_sd=16.1136;
for(i in 1:N){
  dat[i,3]=(dat[i,3]-age_mu)/age_sd
}

randat=dat[sample(1:N,N,replace=FALSE),]
Y=randat[,1]; # Y is SBP
S=t(as.matrix(randat[,3,drop=FALSE])); # S is AGE
for(i in 1:N){S[1,i]=0;}
X=t(as.matrix(randat[,3:5,drop=FALSE])); # X is Age, Sodium, Total_Fat
U=t(as.matrix(randat[,2:5,drop=FALSE])); # U is BMI, Age, Sodium, Total_Fat
s=dim(S)[1];
p1=dim(X)[1];
p2=dim(U)[1]-p1;
slist=c(0);
slistlength=length(slist);

h=0.1;
A=runif(10**8,-1,1);
Knorm=(mean(Ker(A,1)*Ker(A,1))*2)**s; # square of L2 norm
rm(A);


### estimation
# M-fold two stage estimate
hatmu=array(0,c(M,slistlength));
hatrho=array(0,c(M,slistlength));
hattheta=array(0,slistlength);
for(m in 1:M){
  firststageindex=c(c(1:((m-1)*(N/M))),c((m*(N/M)+1):N))
  if(m==1){firststageindex=c(c((m*(N/M)+1):N))}
  if(m==M){firststageindex=c(c(1:((m-1)*(N/M))))}
  secondstageindex=c(((m-1)*(N/M)+1):(m*(N/M)))
  
  U1=U[,firststageindex,drop=FALSE];U2=U[,secondstageindex,drop=FALSE];
  S1=S[,firststageindex,drop=FALSE];S2=S[,secondstageindex,drop=FALSE];
  X1=X[,firststageindex,drop=FALSE];X2=X[,secondstageindex,drop=FALSE];
  Y1=Y[firststageindex];Y2=Y[secondstageindex];
  
  # superleaner for g and d
  learners <- c("SL.glm", "SL.gam", "SL.glmnet.lasso", "SL.nnet")
  sl_model <- SuperLearner(Y = Y1, X = as.data.frame(t(U1)), family = gaussian(), SL.library = learners)
  hatg=predict(sl_model, newdata = as.data.frame(t(U2)))$pred
  Ygs=(Y2-hatg)*(Y2-hatg)
  sl_model <- SuperLearner(Y = Y1, X = as.data.frame(t(X1)), family = gaussian(), SL.library = learners)
  hatd=predict(sl_model, newdata = as.data.frame(t(X2)))$pred
  Yds=(Y2-hatd)*(Y2-hatd)
  
  # local mean for theta
  for(k in c(1:slistlength)){
    s0=slist[k]
    hatmu[m,k]=mean(as.numeric(Ker(S2-s0,h))*Ygs)
    hatrho[m,k]=mean(as.numeric(Ker(S2-s0,h))*Yds)
  }
} # end of m in 1:M
for(k in c(1:slistlength)){
  hattheta[k]=sum(hatmu[,k])/sum(hatrho[,k])
  hattheta[k]=min(max(hattheta[k],0),1)
}
for(k in c(1:slistlength)){
  hattheta[k]=min(max(hattheta[k],0),1)
}


### inference 
  # 2M-fold two stage estimate
  hatmu=array(0,c(2*M,slistlength));
  hatrho=array(0,c(2*M,slistlength));
  fitg=array(0,c(2*M,N));
  fitmu=array(0,c(2*M,N));
  fitd=array(0,c(2*M,N));
  fitrho=array(0,c(2*M,N));
  
  # confidence band
  xi=matrix(rnorm(N*B,1,1), nrow=N, ncol=B)
  cbmu=array(0,c(2*M,slistlength,B));
  cbrho=array(0,c(2*M,slistlength,B));
  # confidence band

  for(m in 1:M){
    firststageindex=c(c(1:((m-1)*(N/(M*2)))),c((m*(N/(M*2))+1):(N/2)))
    if(m==1){firststageindex=c(c((m*(N/(M*2))+1):(N/2)))}
    if(m==M){firststageindex=c(c(1:((m-1)*(N/(M*2)))))}
    secondstageindex=c(((m-1)*(N/(M*2))+1):(m*(N/(M*2))))

    U1=U[,firststageindex,drop=FALSE];U2=U[,secondstageindex,drop=FALSE];
    S1=S[,firststageindex,drop=FALSE];S2=S[,secondstageindex,drop=FALSE];
    X1=X[,firststageindex,drop=FALSE];X2=X[,secondstageindex,drop=FALSE];
    Y1=Y[firststageindex];Y2=Y[secondstageindex];
    
    # superleaner for g 
    learners <- c("SL.glm", "SL.gam", "SL.glmnet.lasso", "SL.nnet")
    sl_model <- SuperLearner(Y = Y1, X = as.data.frame(t(U1)), family = gaussian(), SL.library = learners)
    hatg=predict(sl_model, newdata = as.data.frame(t(U2)))$pred
    Ygs=(Y2-hatg)*(Y2-hatg)
    fitg[m,]=predict(sl_model, newdata = as.data.frame(t(U)))$pred

    for(k in c(1:slistlength)){
      s0=slist[k]
      hatmu[m,k]=mean(as.numeric(Ker(S2-s0,h))*Ygs)
    }
    for(i in 1:N){
      fitmu[m,i]=mean(as.numeric(Ker(S2-S[i],h))*Ygs)
    }
    
    # confidence band
    for(k in 1:slistlength){
      for(b in 1:B){
        s0=slist[k]
        cbmu[m,k,b]= sum(xi[secondstageindex,b]*as.numeric(Ker(S2-s0,h))*Ygs)/sum(xi[secondstageindex,b])
      }
    }
    # confidence band
    
  } # end of m in 1:M
  
  for(m in (M+1):(M*2)){
    firststageindex=c(c((N/2+1):((m-1)*(N/(M*2)))),c((m*(N/(M*2))+1):N))
    if(m==(M+1)){firststageindex=c(c((m*(N/(M*2))+1):N))}
    if(m==(M*2)){firststageindex=c(c((N/2+1):((m-1)*(N/(M*2)))))}
    secondstageindex=c(((m-1)*(N/(M*2))+1):(m*(N/(M*2))))

    U1=U[,firststageindex,drop=FALSE];U2=U[,secondstageindex,drop=FALSE];
    S1=S[,firststageindex,drop=FALSE];S2=S[,secondstageindex,drop=FALSE];
    X1=X[,firststageindex,drop=FALSE];X2=X[,secondstageindex,drop=FALSE];
    Y1=Y[firststageindex];Y2=Y[secondstageindex];
    
    # superleaner for d
    learners <- c("SL.glm", "SL.gam", "SL.glmnet.lasso", "SL.nnet")
    sl_model <- SuperLearner(Y = Y1, X = as.data.frame(t(X1)), family = gaussian(), SL.library = learners)
    hatd=predict(sl_model, newdata = as.data.frame(t(X2)))$pred
    Yds=(Y2-hatd)*(Y2-hatd)
    fitd[m,]=predict(sl_model, newdata = as.data.frame(t(X)))$pred
    
    for(k in c(1:slistlength)){
      s0=slist[k]
      hatrho[m,k]=mean(as.numeric(Ker(S2-s0,h))*Yds)
    }
    for(i in 1:N){
      fitrho[m,i]=mean(as.numeric(Ker(S2-S[i],h))*Yds)
    }
    
    # confidence band
    for(k in 1:slistlength){
      for(b in 1:B){
        s0=slist[k]
        cbrho[m,k,b]= sum(xi[secondstageindex,b]*as.numeric(Ker(S2-s0,h))*Yds)/sum(xi[secondstageindex,b])
      }
    }
    # confidence band
    
  } # end of m in (M+1):(M*2)
  
  # hatthetastar
  hatthetastar=array(0,slistlength);
  for(k in c(1:slistlength)){
    hatthetastar[k]=sum(hatmu[,k])/sum(hatrho[,k])
  }
  for(k in c(1:slistlength)){
    hatthetastar[k]=min(max(hatthetastar[k],0),1)
  }
  
  # hatV
  averhatmu=array(0,c(slistlength));
  averhatrho=array(0,c(slistlength));
  for(k in 1:slistlength){
    averhatmu[k]=sum(hatmu[,k])/M;
    averhatrho[k]=sum(hatrho[,k])/M;
  }
  
  averfitg=array(0,c(N));
  averfitmu=array(0,c(N));
  averfitd=array(0,c(N));
  averfitrho=array(0,c(N));
  for(i in 1:N){
    averfitg[i]=sum(fitg[,i])/M
    averfitmu[i]=sum(fitmu[,i])/M
    averfitd[i]=sum(fitd[,i])/M
    averfitrho[i]=sum(fitrho[,i])/M
  }
  
  hatV1=array(0,slistlength);
  hatV2=array(0,slistlength);
  hatV=array(0,slistlength);
  radius=array(0,slistlength);
  ciupper=array(0,slistlength);
  cilower=array(0,slistlength);
  cilength=array(0,slistlength);
  for(k in 1:slistlength){
    s0=slist[k]
    temp1=(Y-averfitg)*(Y-averfitg)-averfitmu
    hatV1[k]=mean(as.numeric(Ker(S-s0,h))*temp1*temp1)
    temp2=(Y-averfitd)*(Y-averfitd)-averfitrho
    hatV2[k]=mean(as.numeric(Ker(S-s0,h))*temp2*temp2)
    hatV[k]=2*Knorm*( hatV1[k]/(averhatrho[k]*averhatrho[k]) + (hatV2[k]*averhatmu[k]*averhatmu[k])/(averhatrho[k]*averhatrho[k]*averhatrho[k]*averhatrho[k]) )
    radius[k]=sqrt(hatV[k]/(N*h))*qnorm(1-tau/2,0,1)
    ciupper[k]=max(min(hatthetastar[k]+radius[k],1),0)
    cilower[k]=min(max(hatthetastar[k]-radius[k],0),1)
    cilength[k]=ciupper[k]-cilower[k]
  }
  
               

