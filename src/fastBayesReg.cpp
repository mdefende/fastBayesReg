#include <RcppArmadillo.h>

// [[Rcpp::depends(RcppArmadillo)]]
using namespace Rcpp;
using namespace arma;

//'@importFrom Rcpp evalCpp
//'@useDynLib fastBayesReg, .registration=TRUE


//'@title Simulate data from the logistic regression model
//'@param n sample size
//'@param p number of candidate predictors
//'@param q number of nonzero predictors
//'@param beta_size effect size of beta coefficients
//'@return a list objects consisting of the following components
//'\describe{
//'\item{delta}{vector of n outcome variables}
//'\item{X}{n x p matrix of candidate predictors}
//'\item{betacoef}{vector of p regression coeficients}
//'\item{R2}{R-squared indicating the proportion of variation explained by the predictors}
//'}
//'@author Jian Kang <jiankang@umich.edu>
//'@examples
//'dat <- sim_linear_reg(100,100,0.5,c(1,-1,1))
//'@export
// [[Rcpp::export]]
Rcpp::List sim_linear_reg(int n = 100, int p = 10, int q = 5,
                          double R2 = 0.95, double X_cor = 0.5,
                          double beta_size = 1){
	arma::mat x = arma::randn<arma::mat>(n,p);
	arma::vec z = arma::randn<arma::vec>(n);
	arma::beta_nonzero =
	for(arma::uword i=0;i<x.n_cols;i++){
		x.col(i) += z;
	}
	arma::vec y = x.cols(0,q-1L)*beta_nonzero;
	double var_y = arma::var(y);
	double sigma2 = var_y*(1.0 - R2)/R2;
	y += arma::randn(n)*sqrt(sigma2);
	return Rcpp::List::create(Named("x") = x,
                           Named("y") = y,
                           Named("sigma2") = sigma2,
                           Named("beta_nonzero") = beta_nonzero);
}

