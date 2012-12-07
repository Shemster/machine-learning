H<-matrix(readBin("histograms.bin", "double", 640000), 40000, 16)

# add small constant to empty bins
eps<-0.01
for (i in 1:dim(H)[1]){
	for (j in 1:dim(H)[2]){
		if (H[i,j]==0){
			H[i,j]<-eps
		}
	}
}

# parameters
k_1<-3
k_2<-4
k_3<-5
t<-0.01

# EM algorithm
MultinomialEM<-function(H,k,t){
	n<-dim(H)[1]
	d<-dim(H)[2]
	theta<-H[sample(1:n,k),]
	# normalization
	theta<-t(theta/sqrt(rowSums(theta*theta)))
	count<-1
	repeat {
		if(count==1){
			a_old<-matrix(0,n,k)
		}
		else{
			a_old<-a
		}
		phi<-exp(H%*%(log(theta))) # NA Inf
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

plot<-function(m){
	m<-matrix(m,nrow=200,ncol=200) #byrow=TRUE
	image(m,col=c("black","grey","white"))
}