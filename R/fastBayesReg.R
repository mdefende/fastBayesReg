# Fast Bayesian Regression Models
#

#'@title Simulate data from the logistic regression model
#'@param n sample size
#'@param p number of candidate predictors
#'@param q number of nonzero predictors
#'@param beta_size effect size of beta coefficients
#'@return a list objects consisting of the following components
#'\describe{
#'\item{delta}{vector of n outcome variables}
#'\item{X}{n x p matrix of candidate predictors}
#'\item{betacoef}{vector of p regression coeficients}
#'\item{prob}{vector of n outcome probabilities}
#'\item{R2}{R-squared indicating the proportion of variation explained by the predictors}
#'}
#'@author Jian Kang <jiankang@umich.edu>
#'@examples
#'dat <- sim_logit_reg_R()
#'@export
sim_logit_reg_R <- function(n=100, p=10, q = 5, beta_size=1){
	X = matrix(rnorm(n*p),nrow=n,ncol=p)
	betacoef <- c(rep(beta_size*c(1,-1),length = q),rep(0,length=p-q))
	Xbeta <- X%*%betacoef
	beta0 = -mean(Xbeta)
	prob = 1/(1+exp(-beta0-Xbeta))
	delta <- as.numeric(runif(n) < prob)
	R2 = var(prob)/var(delta)
	return(list(delta=delta,X=X,betacoef=betacoef, beta0 = beta0,prob=prob,R2 = R2))
}

#'@export
#'@importFrom glmnet glmnet
wrap_glmnet <- function(y,X,alpha=1,family="gaussian",intercept=FALSE,...){
	elapsed <- proc.time()[3]
	cv_res <- glmnet::cv.glmnet(X,y,alpha=alpha,family=family,...)
	res <- glmnet::glmnet(X,y,alpha=alpha,lambda=cv_res$lambda.1se,family=family,...)
	if(family=="multinomial"){
		betacoef = sapply(1:length(res$beta),function(i) as.matrix(res$beta[[i]]))
	} else{
		betacoef = as.numeric(res$beta)
	}
	elapsed <- proc.time()[3] - elapsed
	return(list(betacoef = betacoef,elapsed=elapsed,glmnet_fit=res))
}


#'Quick call the horseshoe function in horseshoe package
#'@param y vector of n outcome variables
#'@param X n x p matrix of candidate predictors
#'@param method.tau method for handling \eqn{\tau}. Select "truncatedCauchy" for full Bayes with the Cauchy prior truncated to \eqn{[1/p, 1]}, "halfCauchy" for full Bayes with the half-Cauchy prior, or "fixed" to use a fixed value (an empirical Bayes estimate, for example).
#'@param burn number of iterations before start to save
#'@param nmc number of MCMC iterations saved
#'@param thin number of iterations to skip between two saved iterations
#'@param method.sigma method for handling \eqn{\sigma^{2}}{\sigma^2}. Select "Jeffreys" for full Bayes with Jeffrey's prior on the error variance \eqn{\sigma^{2}}{\sigma^2}, or "fixed" to use a fixed value (an empirical Bayes estimate, for example).
#'@param ... other parameters (see \link{horseshoe})
#'@return a list of object consisting of two components
#'\describe{
#'\item{betacoef}{a vector of posterior mean of \eqn{p} regression coeficients}
#'\item{elapsed}{running time}
#'}
#'@author Jian Kang <jiankang@umich.edu>
#'@examples
#'dat <- sim_linear_reg(n=200,p=2000,X_cor=0.9,q=6)
#'res <- with(dat,wrap_horseshoe(y,X))
#'print(comp_sparse_SSE(dat$betacoef,res$betacoef))
#'print(res$elapsed)
#'@export
#'@importFrom horseshoe horseshoe
wrap_horseshoe <- function(y,X,method.tau="halfCauchy",
													 burn=500,nmc=500,thin=1,method.sigma="Jeffreys",...){
	elapsed <- proc.time()[3]
	hsres <- horseshoe::horseshoe(y, X, method.tau = method.tau,
						 method.sigma = method.sigma,
						burn = burn, nmc = nmc, thin = thin,...)
	elapsed <- proc.time()[3] - elapsed
	return(list(betacoef = hsres$BetaHat,elapsed=elapsed,hsres=hsres))
}

#'Compute the sum of squared errors for sparse regression coefficients
#'@param truebeta a vector of true regression coefficients
#'@param estbeta a vector of estimated regression coefficients
#'@return a vector of three measures
#'\describe{
#'\item{overall}{Sum of squared errors for all regression coefficients}
#'\item{nonzero}{Sum of squared errors for nonzero regression coefficients}
#'\item{zero}{Sum of squared errors for zero regression coefficients}
#'}
#'@author Jian Kang <jiankang@umich.edu>
#'@examples
#'dat <- sim_linear_reg(n=2000,p=200,X_cor=0.9,q=6)
#'res <- with(dat1,fast_normal_lm(y,X))
#'print(comp_sparse_SSE(dat$betacoef,res$post_mean$betacoef))
#'@export
comp_sparse_SSE <- function(truebeta,estbeta){
	overall <- sum((truebeta - estbeta)^2)
	nonzero <- 0
	zero <- 0
	nonzero_idx <- which(truebeta!=0)
	if(length(nonzero_idx)>0){
		nonzero <- sum((truebeta[nonzero_idx] - estbeta[nonzero_idx])^2)
	}
	zero_idx <- which(truebeta==0)
	if(length(zero_idx)>0){
		zero <- sum((truebeta[zero_idx] - estbeta[zero_idx])^2)
	}
	return(c(overall=overall,nonzero=nonzero,zero=zero))
}

#'Evaluate classification accuracy
#'@param pred a vector of predicted class indicators, taking two distict values, e.g. 0,1
#'@param obs a vector of observed class indicators, taking two distict values, e.g. 0,1
#'@param levels indicator class values, default c(1,0)
#'@return a vector of five elements TPR, FPR, FDR, ACC, BA
#'\describe{
#'\item{TPR}{True positive rate}
#'\item{FPR}{False positive rate}
#'\item{FDR}{False discovery rate}
#'\item{ACC}{Accuracy}
#'\item{BA}{Balanced Accuracy}
#'}
#'@author Jian Kang <jiankang@umich.edu>
#'@examples
#'dat <- sim_logit_reg(n=2000,p=20,X_cor=0.9,q=6)
#'res <- with(dat,fast_normal_logit(y,X))
#'res_glmnet <- with(dat,wrap_glmnet(y,X,family=binomial()))
#'tab = rbind(comp_class_acc(as.numeric(res$post_mean$prob>0.5),dat$y),
#'comp_class_acc(as.numeric(predict(res_glmnet$glmnet_fit,dat$X,type = "response")>0.5),dat$y))
#'rownames(tab) = c("Bayes","glmnet")
#'print(tab)
#'@export
comp_class_acc <- function(pred,obs,levels=c(1,0)){
	f_pred = factor(pred,levels=levels)
	f_obs = factor(obs,levels=levels)
	tab = table(f_obs,f_pred)
	TP = tab[1,1]
	FN = tab[1,2]
	FP = tab[2,1]
	TN = tab[2,2]
	P = TP + FN
	N = TN + FP
	PP = TP + FP

	if(P>0)
		TPR = TP/P
	else
		TPR = 0
	if(N>0)
		FPR = FP/N
	else
		FPR = 0

	if(PP>0)
		FDR = FP/PP
	else
		FDR = 0

	ACC = (TP+TN)/(TP+TN+FP+FN)

	return(c(TPR=TPR,FPR=FPR, FDR=FDR,ACC=ACC,
					 BA = (TPR+(1-FPR))*0.5))
}
