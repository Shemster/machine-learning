library(ggplot2)

# generate training data
n<-25
train_x<-runif(n,min=0,max=1)
eps<-rnorm(n,mean=0,sd=0.1)
train_y<-matrix((sin(3*train_x)/(3*train_x))+eps, ncol=1)

# parameters
d<-21
mu<-runif(d,min=0,max=1)
sigma<-0.04
var<-sigma^2
lambda_1<-0.001
lambda_2<-0.05
lambda_3<-5

# non-linear mapping of data
hmap<-function(x,means){
	h<-matrix(NA,nrow=length(x),ncol=length(means))
	for (i in 1:length(x)){
		hrow<-vector()
		for (j in 1:length(means)){
			hrow<-c(hrow,exp(-(x[i]-means[j])^2/(2*var)))
		}
		h[i,]<-hrow
	}
	h<-as.matrix(cbind(1,h))
	return(h)
}

# regression
regress_gauss<-function(x,y,means,var,lambda){
	h<-hmap(x,means)
	beta<-solve(t(h)%*%h+lambda*(diag(x=1,d+1,d+1)))%*%t(h)%*%y
	h_beta<-rbind(h,t(beta))
	return(h_beta)
}

# plot of estimated regression function
plot_func<-function(lambda){
	test_x<-seq(0,1,length.out=n)

	# function estimated from training vector train_x
	h_beta<-regress_gauss(train_x,train_y,mu,var,lambda)
	beta<-h_beta[(nrow(h_beta)),]
	h<-hmap(test_x,mu)
	y<-h%*%beta

	# function estimated by beta_avg by repeating training process 100 times
	times<-100
	beta_avg<-matrix(NA,nrow=(d+1),ncol=times)
	for (i in 1:times){
		train_x<-runif(n,min=0,max=1)
		eps<-rnorm(n,mean=0,sd=0.1)
		train_y<-matrix((sin(3*train_x)/(3*train_x))+eps, ncol=1)
		h_beta<-regress_gauss(train_x,train_y,mu,var,lambda)
		beta_avg[,i]<-h_beta[(nrow(h_beta)),]
	}
	beta_avg<-rowMeans(beta_avg)
	y_avg<-h%*%beta_avg

	# plot two estimated functions
	data<-data.frame(test_x,y,y_avg)
	pdf(paste('estimated_functions_lambda_',lambda,'.pdf',sep=''))
	plot<-ggplot(data, aes(test_x))+geom_line(aes(y=y, colour="predicted"))+geom_line(aes(y=y_avg, colour="predicted_avg"))+xlab("test data")+ylab("predicted value")+scale_colour_discrete(name = "Estimated Functions")
	print(plot)
	dev.off()
}

# plot of estimation error
plot_err<-function(){
	lambdas<-exp(seq(-5,2,by=0.3))
	errs<-rep(NA,length(lambdas))
	for (l in 1:length(lambdas)){
		times<-100
		beta_avg<-matrix(NA,nrow=(d+1),ncol=times)
		for (j in 1:times){
			train_x<-runif(n,min=0,max=1)
			eps<-rnorm(n,mean=0,sd=0.1)
			train_y<-matrix((sin(3*train_x)/(3*train_x))+eps, ncol=1)
			h_beta<-regress_gauss(train_x,train_y,mu,var,lambdas[l])
			beta_avg[,j]<-h_beta[(nrow(h_beta)),]
		}
		beta_avg<-rowMeans(beta_avg)
		m<-100
		z<-seq(0, 1, length.out=m)
		eps_z<-rnorm(m,mean=0,sd=0.1)
		f_z<-matrix((sin(3*z)/(3*z))+eps_z, ncol=1)
		f_z[1,]<-1+eps_z[1]
		h_z<-hmap(z,mu)
		f_hat_z<-h_z%*%beta_avg
		errs[l]<-colMeans((f_z-f_hat_z)*(f_z-f_hat_z))
	}
	data<-data.frame(lambdas,errs)
	pdf('error_lambda.pdf')
	plot<-ggplot(data, aes(lambdas))+geom_line(aes(y=errs))+xlab("Lambda")+ylab("Averaged Error")
	print(plot)
	dev.off()
	pdf('error_lambda_zero.pdf')
	plot2<-ggplot(data, aes(lambdas))+geom_line(aes(y=errs))+xlab("Lambda")+ylab("Averaged Error")+scale_x_continuous(limits=c(0,1))+scale_y_continuous(limits=c(0,0.04))
	print(plot2)
	dev.off()
}