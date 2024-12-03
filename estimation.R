library("foreach")
library("doParallel")
require("ggplot2")
library("expm")
library("mvtnorm")

#kernel
Ker=function(s,h){
  temp=ifelse((s-h)>0,0,ifelse((s+h)<0,0,((1-(abs(s/h))**3)**3)*70/(h*81)))
  return(temp);
}


mento=function(it){
  library("SuperLearner")
  library("randomForest")
  library("glmnet")
  library("nnet")
  library("foreach")
  library("doParallel")
  require("ggplot2")
  library("expm")
  library("mvtnorm")
  SL.glmnet.lasso <- function(...) {
    SL.glmnet(..., alpha = 1)
  }
  
  # generate data
  if(scorenumber==1){
    U=matrix(runif(N*(p1+p2),-sqrt(3),sqrt(3)), nrow=p1+p2, ncol=N)
  }
  if(scorenumber==2){
    U=t(rmvnorm(N, as.numeric(array(0,p1+p2)), Sigma))
  }
  S=U[1:s,,drop=FALSE]
  W=U[(1+s):p1,,drop=FALSE]
  X=U[1:p1,,drop=FALSE]
  Z=U[(p1+1):(p1+p2),,drop=FALSE]
  
  Y=array(0,N);
  for(i in 1:N){
    Y[i]=sum(beta1*S[,i])+sum(beta2*W[,i])+sum(beta3*Z[,i])+rnorm(1,0,1)*((S[1,i]+5)/3);
  }
  Y=as.numeric(Y)
  h=1.06*sd(S)*(N**(-1/5))
  
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
    W1=W[,firststageindex,drop=FALSE];W2=W[,secondstageindex,drop=FALSE];
    X1=X[,firststageindex,drop=FALSE];X2=X[,secondstageindex,drop=FALSE];
    Z1=Z[,firststageindex,drop=FALSE];Z2=Z[,secondstageindex,drop=FALSE];
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
  }
  for(k in c(1:slistlength)){
    hattheta[k]=min(max(hattheta[k],0),1)
  }
  
  
  return(as.list(hattheta))
}



#################################### main function
M=5;
s=1;
slist=seq(-1.5,1.5,0.1);
slistlength=length(slist);

for(N in c(2000,5000,10000)){
for(scorenumber in c(1:2)){
for(p1 in c(2,5,10,20)){
for(beta3value in c(0,0.5,1)){
p2=p1;
beta1=array(1,s);
beta2=array(1,p1-s);
beta3=array(beta3value,p2);
if(p1>5){for(k in 5:(p1-s)){beta2[k]=0;}}
if(p2>5){for(k in 6:p2){beta3[k]=0;}}

Sigma=array(0,c(p1+p2,p1+p2)); # variance matrix for normal distribution
for(i in c(1:(p1+p2))){for(j in c(1:(p1+p2))){Sigma[i,j]=0.2**(abs(i-j));}}

itermax=1000;
closeAllConnections();
closeAllConnections();
cl <- makeCluster(100);
registerDoParallel(cl);
result <- foreach(it=c(1:itermax), .combine='c') %dopar% mento(it);
stopCluster(cl);
closeAllConnections();
closeAllConnections();

hattheta=array(0,c(slistlength,itermax));

for(it in 1:itermax){
  for(k in c(1:slistlength)){
    hattheta[k,it]=result[[slistlength*(it-1)+k]]
  }
}

realtheta=array(0,slistlength)
for(k in c(1:slistlength)){
  s0=slist[k]
  if(scorenumber==1){
    realtheta[k]= (((5+s0)/3)*((5+s0)/3)) / (((5+s0)/3)*((5+s0)/3)+min(p2,5)*beta3value*beta3value)
  }
  if(scorenumber==2){
    Sigma11=Sigma[1:p1,1:p1,drop=FALSE]
    Sigma12=Sigma[1:p1,(1+p1):(p1+p2),drop=FALSE]
    Sigma21=Sigma[(1+p1):(p1+p2),1:p1,drop=FALSE]
    Sigma22=Sigma[(1+p1):(p1+p2),(1+p1):(p1+p2),drop=FALSE]
    temp=c(-t(as.matrix(beta3))%*%Sigma21%*%solve(Sigma11),beta3)
    a=temp[1]
    b=as.matrix(temp[2:(p1+p2)])
    Sigma11=Sigma[1,1,drop=FALSE]
    Sigma12=Sigma[1,2:(p1+p2),drop=FALSE]
    Sigma21=Sigma[2:(p1+p2),1,drop=FALSE]
    Sigma22=Sigma[2:(p1+p2),2:(p1+p2),drop=FALSE]
    zvar=(a*s0+t(b)%*%Sigma21%*%solve(Sigma11)%*%s0)**2 + t(b)%*%(Sigma22-Sigma21%*%solve(Sigma11)%*%Sigma12)%*%b
    realtheta[k]= (((5+s0)/3)*((5+s0)/3)) / (((5+s0)/3)*((5+s0)/3)+zvar)
  }
}

superror=array(0,c(itermax))
for(it in c(1:itermax)){
  superror[it]=max(abs(hattheta[,it]-realtheta))
}

output=c( N,scorenumber,p1,beta3value,
          round(mean( abs(hattheta[6,]-realtheta[6])   , na.rm=TRUE),4),
          round(sd(   abs(hattheta[6,]-realtheta[6])   , na.rm=TRUE),4),
          round(mean( abs(hattheta[16,]-realtheta[16]) , na.rm=TRUE),4),
          round(sd(   abs(hattheta[16,]-realtheta[16]) , na.rm=TRUE),4),
          round(mean( abs(hattheta[26,]-realtheta[26]) , na.rm=TRUE),4),
          round(sd(   abs(hattheta[26,]-realtheta[26]) , na.rm=TRUE),4),
          round(mean( superror                         , na.rm=TRUE),4),
          round(sd(   superror                         , na.rm=TRUE),4),
          ""
)
cat(output,'\n');


}}}}
