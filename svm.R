library('e1071')
library('scatterplot3d')

uspsdata<-read.table("./uspsdata/uspsdata.txt")
label<-read.table("./uspsdata/uspscl.txt")

# set.seed(12)
# set.seed(6)
# set.seed(18)
set.seed(24)
test<-sort(sample(1:nrow(uspsdata),nrow(uspsdata)*0.2))
testdata<-uspsdata[test,]
testlabel<-label[test,]
traindata<-uspsdata[!(1:nrow(uspsdata)) %in% test,]
trainlabel<-label[!(1:nrow(label)) %in% test,]

cost<-2^seq(-5,5,by=0.5)
# n<-length(cost)
# linear_error<-rep(NA,n)

# for (i in 1:n){
# 	model<-svm(traindata,trainlabel,type='C',kernel='linear',cost=cost[i],cross=5)
# 	linear_error[i]<-1-model$tot.accuracy/100
# }

# pdf("linear_plot.pdf")
# linear_plot<-plot(cost,linear_error,xlab="margin parameter",ylab="misclassification rate",col=rgb(255,0,0,80,maxColorValue=255))
# dev.off()

# tune_linear<-tune.svm(traindata,trainlabel,kernel='linear',cost=cost,tunecontrol=tune.control(cross=5))
# best_linear_cost<-tune_linear$best.parameters[[1]]
# print(tune_linear$best.parameters)
# linear_model<-svm(traindata,trainlabel,type='C',kernel='linear',cost=best_linear_cost,cross=5)
# linear_pred<-predict(linear_model,testdata)
# linear_test_table<-table(linear_pred,testlabel)
# linear_model_error<-(length(testlabel)-sum(diag(linear_test_table)))/(length(testlabel))
# print(linear_model_error)

# linear_dif<-as.integer(as.vector(linear_pred))-testlabel
# linear_missed<-list()
# count<-1
# for (i in 1:length(linear_dif)){
# 	if (linear_dif[i]!=0){
# 		linear_missed[[count]]<-testdata[i,]
# 		count<-count+1
# 		if (count==6){
# 			break
# 		}
# 	}
# }
# for (i in 1:length(linear_missed)){
# 	m<-matrix(as.vector(as.numeric(linear_missed[[i]])),ncol=16,nrow=16,byrow=TRUE)
# 	png(paste('linear_missed',i,'.png',sep=''))
# 	img<-image(m,axes=FALSE,col=grey(seq(0,1,length=256)))
# 	dev.off()
# }

# count<-1
# for (i in 1:length(linear_dif)){
# 	if (linear_dif[i]==0){
# 		m<-matrix(as.vector(as.numeric(testdata[i,])),ncol=16,nrow=16,byrow=TRUE)
# 		png(paste('linear_correct',i,'.png',sep=''))
# 		img<-image(m,axes=FALSE,col=grey(seq(0,1,length=256)))
# 		dev.off()
# 		count<-count+1
# 		if (count==6){
# 			break
# 		}
# 	}
# }

gamma<-10^seq(-4,6,by=0.5)
# m<-length(gamma)
# param<-expand.grid(cost,gamma)
# p<-length(param)
# rbf_error<-rep(NA,p)
# l<-1
# for (j in 1:n){
# 	for (k in 1:m){
# 		model<-svm(traindata,trainlabel,type='C',kernel='radial',cost=cost[j],gamma=gamma[k],cross=5)
# 		rbf_error[l]<-1-model$tot.accuracy/100
# 		l<-l+1
# 	}
# }

# plot_data<-data.frame(param,rbf_error)
# names(plot_data)[1]<-"cost"
# names(plot_data)[2]<-"gamma"
# pdf("rbf_plot.pdf")
# rbf_plot<-scatterplot3d(plot_data$cost,plot_data$gamma,plot_data$rbf_error,xlab="margin parameter",ylab="bandwidth",zlab="misclassification rate",type="h",color=(rgb(255,0,0,50,max=255)))
# dev.off()

tune_rbf<-tune.svm(traindata,trainlabel,kernel='radial',cost=cost,gamma=gamma,tunecontrol=tune.control(cross=5))
best_rbf_gamma<-tune_rbf$best.parameters[[1]]
best_rbf_cost<-tune_rbf$best.parameters[[2]]
# print(tune_rbf$best.parameters)
rbf_model<-svm(traindata,trainlabel,type='C',kernel='radial',cost=best_rbf_cost,gamma=best_rbf_gamma,cross=5)
rbf_pred<-predict(rbf_model,testdata)
# rbf_test_table<-table(rbf_pred,testlabel)
# rbf_model_error<-(length(testlabel)-sum(diag(rbf_test_table)))/(length(testlabel))
# print(rbf_model_error)

rbf_dif<-as.integer(as.vector(rbf_pred))-testlabel
rbf_missed<-list()
count<-1
for (i in 1:length(rbf_dif)){
	if (rbf_dif[i]!=0){
		rbf_missed[[count]]<-testdata[i,]
		count<-count+1
		if (count==6){
			break
		}
		
	}
}
for (i in 1:length(rbf_missed)){
	m<-matrix(as.vector(as.numeric(rbf_missed[[i]])),ncol=16,nrow=16,byrow=TRUE)
	png(paste('seed24_rbf_missed',i,'.png',sep=''))
	img<-image(m,axes=FALSE,col=grey(seq(0,1,length=256)))
	dev.off()
}

# count<-1
# for (i in 1:length(rbf_dif)){
# 	if (rbf_dif[i]==0){
# 		m<-matrix(as.vector(as.numeric(testdata[i,])),ncol=16,nrow=16,byrow=TRUE)
# 		png(paste('rbf_correct',i,'.png',sep=''))
# 		img<-image(m,axes=FALSE,col=grey(seq(0,1,length=256)))
# 		dev.off()
# 		count<-count+1
# 		if (count==6){
# 			break
# 		}
		
# 	}
# }

system('for f in *.png; do convert $f -rotate 90 $f; done')