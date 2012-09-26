normalize<-function(v){
	sum<-0
	for (i in 1:length(v)){
		sum<-sum+(v[i])^2
	}
	v<-v/sqrt(sum)
	return(v)
}

fakedata<-function(a,n_samples){
	normalize(a)
	coor<-list()
	lab<-list()
	d<-length(a)-1
	for (i in 1:n_samples) {
		x_i<-c(rnorm(d),1)
		coor[[i]]<-x_i
		lab[[i]]<-sign(crossprod(a,x_i))
	}
	data_set<-list("data"=coor, "label"=lab)
	return(data_set)
}

classify<-function(S,z){
	lab<-list()
	for (i in 1:length(S)){
		lab[[i]]<-sign(crossprod(S[[i]],z))
	}
	return(lab)
}

batch_perceptrain<-function(S,y){
	d<-length(S[[1]])
	z<-rep(1,d)
	z_history<-list(z)
	lab<-list()
	k<-1
	while (TRUE) {
		lab<-classify(S,z)
		cost<-0
		cost_grad<-rep(0,d)
		for (i in 1:length(lab)) {
			if (lab[[i]]!=y[[i]]) {
				cost<-(cost+abs(sum(S[[i]]*z)))
				cost_grad<-(cost_grad-y[[i]]*S[[i]])
			}
		}
		if (cost==0){
			break
		}
		else {
			k<-k+1
			a<-1/k
			z<-z-a*cost_grad
			z_history[[length(z_history)+1]]<-z
		}
	}
	return (z_history)
}

fiss_perceptrain<-function(S,y){
	d<-length(S[[1]])
	z<-rep(1,d)
	z_history<-list(z)
	lab<-list()
	k<-1
	while (TRUE) {
		lab<-classify(S,z)
		cost<-0
		for (i in 1:length(S)) {
			if (lab[[i]]!=y[[i]]) {
				cost<-cost+abs(sum(S[[i]]*z))
				cost_grad<-y[[i]]*S[[i]]
			}
		}
		if (cost==0) {
			break
		}
		else {
			k<k+1
			z<-z+cost_grad
			z_history[[length(z_history)+1]]<-z
		}
	}
	return (z_history)
}

matrixconv<-function(list){
	row=length(list)
	col=length(list[[1]])
	m<-matrix(NA, row, col)
	for (i in 1:row) {
		m[i,]<-list[[i]]
	}
	return (m)
}

plot_train<-function(train_data,z_his){
	train_data<-matrixconv(train_data)
	for (i in 1:length(z_his)){
		z_his[[i]]<-normalize(z_his[[i]])
	}
	z_his<-matrixconv(z_his)
	plot(train_data[,1],train_data[,2], pch=4, col=4, cex=1.5, xlab="x", ylab="y")
	points(z_his[,1],z_his[,2], pch=2,col=3, type="o")
}

plot_test<-function(test_data,zvector){
	test_data<-matrixconv(test_data)
	plot(test_data[,1],test_data[,2], pch=4, col=4, cex=1.5, xlab="x", ylab="y")
	abline(-zvector[1]/zvector[2],0, col=2)
}