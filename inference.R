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
  h=1.06*sd(S)*(N**(-1/5));
  
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
    W1=W[,firststageindex,drop=FALSE];W2=W[,secondstageindex,drop=FALSE];
    X1=X[,firststageindex,drop=FALSE];X2=X[,secondstageindex,drop=FALSE];
    Z1=Z[,firststageindex,drop=FALSE];Z2=Z[,secondstageindex,drop=FALSE];
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
    W1=W[,firststageindex,drop=FALSE];W2=W[,secondstageindex,drop=FALSE];
    X1=X[,firststageindex,drop=FALSE];X2=X[,secondstageindex,drop=FALSE];
    Z1=Z[,firststageindex,drop=FALSE];Z2=Z[,secondstageindex,drop=FALSE];
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
  for(k in 1:slistlength){
    s0=slist[k]
    temp1=(Y-averfitg)*(Y-averfitg)-averfitmu
    hatV1[k]=mean(as.numeric(Ker(S-s0,h))*temp1*temp1)
    temp2=(Y-averfitd)*(Y-averfitd)-averfitrho
    hatV2[k]=mean(as.numeric(Ker(S-s0,h))*temp2*temp2)
    hatV[k]=2*Knorm*( hatV1[k]/(averhatrho[k]*averhatrho[k]) + (hatV2[k]*averhatmu[k]*averhatmu[k])/(averhatrho[k]*averhatrho[k]*averhatrho[k]*averhatrho[k]) )
    radius[k]=sqrt(hatV[k]/(N*h))*qnorm(1-tau/2,0,1)
  }
  
  # confidence band
  cbtheta=array(0,c(slistlength,B));
  for(k in 1:slistlength){
    for(b in 1:B){
      s0=slist[k]
      cbtheta[k,b]=sum(cbmu[,k,b])/sum(cbrho[,k,b])
    }
  }
  normalizederror=array(0,c(slistlength,B));
  for(k in 1:slistlength){
    for(b in 1:B){
      normalizederror[k,b]=abs(cbtheta[k,b]-hatthetastar[k])*sqrt((N*h)/hatV[k])
    }
  }
  supnormalizederror=array(0,B);
  for(b in 1:B){
    supnormalizederror[b]=max(normalizederror[,b])
  }
  Qtau=quantile(supnormalizederror, probs = 1-tau)
  cbupper=array(0,slistlength);
  cblower=array(0,slistlength);
  for(k in 1:slistlength){
    cbupper[k]=max(min(hatthetastar[k]+sqrt(hatV[k]/(N*h))*Qtau,1),0)
    cblower[k]=min(max(hatthetastar[k]-sqrt(hatV[k]/(N*h))*Qtau,0),1)
  }
  # confidence band
  
  return(as.list(c(hatthetastar,radius,cbupper,cblower)))
}


#################################### main function
M=5;
s=1;
slist=seq(-1.5,1.5,0.1);
slistlength=length(slist);
tau=0.05;
B=1000;
for(N in c(100000)){
for(scorenumber in c(1:2)){
for(p1 in c(2,5,10,20)){
for(beta3value in c(0,0.5,1)){
p2=p1;
beta1=array(1,s);
beta2=array(1,p1-s);
beta3=array(beta3value,p2);
if(p1>5){for(k in 5:(p1-s)){beta2[k]=0;}}
if(p2>5){for(k in 6:p2){beta3[k]=0;}}

A=runif(10**8,-1,1);
Knorm=(mean(Ker(A,1)*Ker(A,1))*2)**s; # square of L2 norm
rm(A);

Sigma=array(0,c(p1+p2,p1+p2)); # variance matrix for normal distribution
for(i in c(1:(p1+p2))){for(j in c(1:(p1+p2))){Sigma[i,j]=0.2**(abs(i-j));}}

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
radius=array(0,c(slistlength,itermax));
ciupper=array(0,c(slistlength,itermax));
cilower=array(0,c(slistlength,itermax));
cilength=array(0,c(slistlength,itermax));
cover=array(0,c(slistlength,itermax));
cbupper=array(0,c(slistlength,itermax));
cblower=array(0,c(slistlength,itermax));
cblength=array(0,c(slistlength,itermax));
cbcover=array(0,c(slistlength,itermax));
for(it in 1:itermax){
  for(k in c(1:slistlength)){
    hattheta[k,it]=result[[ 4*slistlength*(it-1)+k ]]
    radius[k,it]=result[[ 4*slistlength*(it-1)+slistlength+k ]]
    cbupper[k,it]=result[[ 4*slistlength*(it-1)+2*slistlength+k ]]
    cblower[k,it]=result[[ 4*slistlength*(it-1)+3*slistlength+k ]]
    
    ciupper[k,it]= max(min(hattheta[k,it]+radius[k,it],1),0)
    cilower[k,it]= min(max(hattheta[k,it]-radius[k,it],0),1)
    cilength[k,it]=ciupper[k,it]-cilower[k,it]
    cover[k,it]= (ciupper[k,it]>=realtheta[k]) && (realtheta[k]>=cilower[k,it])
    
    cblength[k,it]=cbupper[k,it]-cblower[k,it]
    cbcover[k,it]= (cbupper[k,it]>=realtheta[k]) && (realtheta[k]>=cblower[k,it])
  }
}
cbaveragelength=array(0,itermax);
cbuniformcover=array(0,itermax);
for(it in 1:itermax){
  cbaveragelength[it]=mean(cblength[,it])
  cbuniformcover[it]=prod(cbcover[,it])
}


output=c( N,scorenumber,p1,beta3value, "   ",
          round(mean( cover[6,], na.rm=TRUE),4)*100,
          round(mean( cilength[6,], na.rm=TRUE),4),
          round(mean( cover[16,], na.rm=TRUE),4)*100,
          round(mean( cilength[16,], na.rm=TRUE),4),
          round(mean( cover[26,], na.rm=TRUE),4)*100,
          round(mean( cilength[26,], na.rm=TRUE),4),
          round(mean( cbuniformcover, na.rm=TRUE),4)*100,
          round(mean( cbaveragelength, na.rm=TRUE),4),
          ""
)
cat(output,'\n');


}}}}
