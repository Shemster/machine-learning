H<-matrix(readBin("histograms.bin", "double", 640000), 40000, 16)

# add small constant to empty bins
eps<-0.01
index<-which(H==0,arr.ind=TRUE)
for (i in 1:dim(index)[1]){
	H[index[i,1],index[i,2]]<-eps
}

# parameters
k_1<-3
k_2<-4
k_3<-5

# EM algorithm
MultinomialEM<-function(H,k,t){
	n<-dim(H)[1]
	d<-dim(H)[2]
	H<-H/rowSums(H) # normalize each histogram to avoid overflow/underflow
	theta<-t(H[sample(1:n,k),])
	count<-1
	repeat {
		if(count==1){
			a_old<-matrix(0,n,k)
		}
		else{
			a_old<-a
		}
		phi<-exp(H%*%(log(theta)))
		a<-phi/colSums(phi)
		b<-t(H)%*%a
		theta<-b/colSums(b)
		c<-norm(a-a_old,"o")
		count<-count+1
		if(c<t)
		break
	}
	m<-max.col(a)
	return(m)
}

# plot to check which parameter produce the best cluster result
tune_par<-function(H,k){
	param<-seq(0.1,0.9,length=9)
	for (i in 1:length(param)){
		m<-MultinomialEM(H,k,param[i])
		m<-matrix(m,nrow=200,ncol=200)
		n<-matrix(NA,nrow=200,ncol=200)
		for (j in 0:(200-1)){
			n[,j+1]<-m[,200-j]
		}
		pdf(paste(k,'_cluster_',param[i],'.pdf',sep=''))
		image(n,axes=FALSE,col=grey(0:k/k))
		dev.off()
	}
}