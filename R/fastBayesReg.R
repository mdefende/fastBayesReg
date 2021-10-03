# Fast Bayesian Regression Models
#

#'Simulate data from the logistic regression model
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
#'dat <- sim_logit_reg()
#'@export
sim_logit_reg <- function(n=100, p=10, q = 5, beta_size=1){
	X = matrix(rnorm(n*p),nrow=n,ncol=p)
	betacoef <- c(rep(beta_size*c(1,-1),length = q),rep(0,length=p-q))
	Xbeta <- X%*%betacoef
	beta0 = -mean(Xbeta)
	prob = 1/(1+exp(-beta0-Xbeta))
	delta <- as.numeric(runif(n) < prob)
	R2 = var(prob)/var(delta)
	return(list(delta=delta,X=X,betacoef=betacoef, beta0 = beta0,prob=prob,R2 = R2))
}


