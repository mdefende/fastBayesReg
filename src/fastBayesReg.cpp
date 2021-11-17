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
//'@param R2 R-squared indicating the proportion of variation explained by the predictors
//'@param beta_size effect size of beta coefficients
//'@param X_cor correlation between covariates
//'@return a list objects consisting of the following components
//'\describe{
//'\item{y}{vector of n outcome variables}
//'\item{X}{n x p matrix of candidate predictors}
//'\item{betacoef}{vector of p regression coeficients}
//'\item{R2}{R-squared indicating the proportion of variation explained by the predictors}
//'\item{sigma2}{noise variance}
//'\item{X_cor}{correlation between covariates}
//'}
//'@author Jian Kang <jiankang@umich.edu>
//'@examples
//'dat<-sim_linear_reg(n=25,p=2000,X_cor=0.9,q=6)
//'summary(with(dat,lm(y~X)))
//'@export
// [[Rcpp::export]]
Rcpp::List sim_linear_reg(int n = 100, int p = 20, int q = 5,
                          double R2 = 0.95, double X_cor = 0.5,
                          double beta_size = 1){
	arma::mat x = sqrt(1.0 - X_cor)*arma::randn<arma::mat>(n,p);
	arma::vec z = sqrt(X_cor)*arma::randn<arma::vec>(n);
	int qh = q/2;
	arma::vec beta_nonzero = arma::repmat(vec({beta_size,-beta_size}),qh,1);
	if(q>2*qh){
		beta_nonzero = arma::join_cols(beta_nonzero,vec({beta_size}));
	}
	for(arma::uword i=0;i<x.n_cols;i++){
		x.col(i) += z;
	}
	arma::vec y = x.cols(0,q-1L)*beta_nonzero;
	double var_y = arma::var(y);
	double sigma2 = var_y*(1.0 - R2)/R2;
	y += arma::randn(n)*sqrt(sigma2);
	return Rcpp::List::create(Named("y") = y,
                           Named("X") = x,
                           Named("betacoef") = arma::join_cols(beta_nonzero,arma::zeros<arma::vec>(p-q)),
                           Named("R2") = R2,
                           Named("sigma2") = sigma2,
                           Named("X_cor") = X_cor);
}

void one_step_update_big_p(arma::vec& betacoef, double& sigma2_eps, double& tau2,
                     double& b_tau, arma::vec& mu, arma::vec& ys,  arma::mat& V, arma::vec& d,arma::vec& d2,
                     arma::vec& y, arma::mat& X,
                      double A2, double a_sigma, double b_sigma,
                     int p, int n){

	arma::vec alpha_1 = arma::randn<arma::vec>(p)*sqrt(sigma2_eps*tau2);
	arma::vec alpha_2 = arma::randn<arma::vec>(n)*sqrt(sigma2_eps);
	arma::vec beta_s = (ys - d%(V.t()*alpha_1) - alpha_2)%d/(1.0 + tau2*d2);
	betacoef = alpha_1 + tau2*V*beta_s;
	mu = X*betacoef;
	arma::vec eps = y - mu;
	double sum_eps2 = arma::accu(eps%eps);
	double sum_beta2 = arma::accu(betacoef%betacoef);
	double inv_tau2 = randg<double>(distr_param((1.0+p)/2.0,1.0/(b_tau+0.5*sum_beta2/sigma2_eps)));
	b_tau = randg<double>(distr_param(1.0,1.0/(1.0/A2 + inv_tau2)));
	tau2 = 1.0/inv_tau2;
	double inv_sigma2_eps = randg<double>(distr_param(a_sigma+(n+p)/2.0, 1.0/(b_sigma+0.5*sum_beta2*inv_tau2+0.5*sum_eps2)));
	sigma2_eps = 1.0/inv_sigma2_eps;
}


void one_step_update_big_n(arma::vec& betacoef, double& sigma2_eps, double& tau2,
                         double& b_tau, arma::vec& mu, arma::vec& ys,  arma::mat& V, arma::vec& d,arma::vec& d2,
                         arma::vec& y, arma::mat& X,
                         double A2, double a_sigma, double b_sigma,
                         int p, int n){

	double inv_tau2 = 1.0/tau2;
	arma::vec alpha_1 = arma::randn<arma::vec>(p)%sqrt(sigma2_eps/(d2 + inv_tau2));
	arma::vec beta_s = d%ys/(d2 + inv_tau2) + alpha_1;
	betacoef = V*beta_s;
	mu = d%beta_s;
	arma::vec eps = ys - mu;
	double sum_eps2 = arma::accu(eps%eps);
	double sum_beta2 = arma::accu(beta_s%beta_s);
	inv_tau2 = randg<double>(distr_param((1.0+p)/2.0,1.0/(b_tau+0.5*sum_beta2/sigma2_eps)));
	b_tau = randg<double>(distr_param(1.0,1.0/(1.0/A2 + inv_tau2)));
	tau2 = 1.0/inv_tau2;
	double inv_sigma2_eps = randg<double>(distr_param(a_sigma+p, 1.0/(b_sigma+0.5*sum_beta2*inv_tau2+0.5*sum_eps2)));
	sigma2_eps = 1.0/inv_sigma2_eps;
}

//'@title Fast Bayesian linear regression with normal priors
//'@param y vector of n outcome variables
//'@param X n x p matrix of candidate predictors
//'@param mcmc_sample number of MCMC iterations saved
//'@param burnin number of iterations before start to save
//'@param thinning number of iterations to skip between two saved iterations
//'@param a_sigma shape parameter in the inverse gamma prior of the noise variance
//'@param b_sigma rate parameter in the inverse gamma prior of the noise variance
//'@param A_tau scale parameter in the half Cauchy prior of the ratio between the coefficient variance and the noise variance
//'@return a list object consisting of two components
//'\describe{
//'\item{post_mean}{a list object of four components for posterior mean statistics}
//'\describe{
//'\item{mu}{a vector of posterior predictive mean of the n training sample}
//'\item{betacoef}{a vector of posterior mean of p regression coeficients}
//'\item{sigma2_eps}{posterior mean of the noise variance}
//'\item{tau2}{posterior mean of the ratio between prior regression coefficient variances and the noise variance}
//'}
//'\item{mcmc}{a list object of three components for MCMC samples}
//'\describe{
//'\item{mu}{a vector of posterior predictive mean of the n training sample}
//'\item{betacoef}{a vector of posterior mean of p regression coeficients}
//'\item{sigma2_eps}{posterior mean of the noise variance}
//'\item{tau2}{posterior mean of the ratio between prior regression coefficient variances and the noise variance}
//'}
//'}
//'@author Jian Kang <jiankang@umich.edu>
//'@examples
//'set.seed(2022)
//'dat1 <- sim_linear_reg(n=2000,p=200,X_cor=0.9,q=6)
//'res1 <- with(dat1,fast_normal_lm(y,X))
//'dat2 <- sim_linear_reg(n=200,p=2000,X_cor=0.9,q=6)
//'res2 <- with(dat2,fast_normal_lm(y,X))
//'tab <- data.frame(rbind(comp_sparse_SSE(dat1$betacoef,res1$post_mean$betacoef),
//'comp_sparse_SSE(dat2$betacoef,res2$post_mean$betacoef)),
//'time=c(res1$elapsed,res2$elapsed))
//'rownames(tab)<-c("n = 2000, p = 200","n = 200, p = 2000")
//'fast_normal_tab <- tab
//'print(fast_normal_tab)
//'@export
// [[Rcpp::export]]
Rcpp::List fast_normal_lm(arma::vec& y, arma::mat& X,
                         int mcmc_sample = 500,
                         int burnin = 500, int thinning = 1,
                         double a_sigma = 0.01, double b_sigma = 0.01,
                         double A_tau = 10){

	arma::wall_clock timer;
	timer.tic();
	arma::vec d;
	arma::mat U;
	arma::mat V;
	arma::svd_econ(U,d,V,X);

	//std::cout << "U (" << U.n_rows << "," << U.n_cols << ")" << std::endl;
	//std::cout << "d (" << d.n_elem  << ")" << std::endl;
	//std::cout << "V (" << V.n_rows << "," << V.n_cols << ")" << std::endl;

	int p = X.n_cols;
	int n = X.n_rows;
	double sigma2_eps = b_sigma/a_sigma;
	double A2 = A_tau*A_tau;
	double b_tau = A2;
	double tau2 = b_tau;
	arma::vec d2 = d%d;
	arma::vec ys = U.t()*y;


	arma::vec betacoef;
	arma::vec mu;

	arma::mat betacoef_list;
	arma::vec sigma2_eps_list;
	arma::vec tau2_list;

	betacoef_list.zeros(p,mcmc_sample);
	sigma2_eps_list.zeros(mcmc_sample);
	tau2_list.zeros(mcmc_sample);


	if(p<n){
		for(int iter=0;iter<burnin;iter++){
			one_step_update_big_n(betacoef, sigma2_eps, tau2,
                 b_tau, mu, ys,  V,  d, d2, y,  X,
                 A2,  a_sigma,  b_sigma, p,  n);
		}
		for(int iter=0;iter<mcmc_sample;iter++){
			for(int j=0;j<thinning;j++){
				one_step_update_big_n(betacoef, sigma2_eps, tau2,
                          b_tau, mu, ys,  V,  d, d2, y,  X,
                          A2,  a_sigma,  b_sigma, p,  n);
			}
			betacoef_list.col(iter) = betacoef;
			sigma2_eps_list(iter) = sigma2_eps;
			tau2_list(iter) = tau2;
		}
	} else{
		for(int iter=0;iter<burnin;iter++){
			one_step_update_big_p(betacoef, sigma2_eps, tau2,
                         b_tau, mu, ys,  V,  d, d2, y,  X,
                         A2,  a_sigma,  b_sigma, p,  n);
		}
		for(int iter=0;iter<mcmc_sample;iter++){
			for(int j=0;j<thinning;j++){
				one_step_update_big_p(betacoef, sigma2_eps, tau2,
                          b_tau, mu, ys,  V,  d, d2, y,  X,
                          A2,  a_sigma,  b_sigma, p,  n);
			}
			betacoef_list.col(iter) = betacoef;
			sigma2_eps_list(iter) = sigma2_eps;
			tau2_list(iter) = tau2;
		}

	}

	betacoef = arma::mean(betacoef_list,1);
	sigma2_eps = arma::mean(sigma2_eps_list);
	tau2 = arma::mean(tau2_list);

	Rcpp::List post_mean = Rcpp::List::create(Named("mu") = X*betacoef,
                                           Named("betacoef") = betacoef,
                                           Named("sigma2_eps") = sigma2_eps,
                                           Named("tau2") = tau2);
	Rcpp::List mcmc = Rcpp::List::create(Named("betacoef") = betacoef_list,
                                      Named("sigma2_eps") = sigma2_eps_list,
                                      Named("tau2") = tau2_list);

	double elapsed = timer.toc();
	return Rcpp::List::create(Named("post_mean") = post_mean,
                           Named("mcmc") = mcmc,
                           Named("elapsed") = elapsed);
}

//'@title Simulate left standard truncated normal distribution
//'@param n sample size
//'@param lower the lower bound
//'@return a vector of random numbers from a left standard truncated
//'normal distribution
//'@author Jian Kang <jiankang@umich.edu>
//'@examples
//'r0 <- rand_left_trucnorm0(1000, lower=2)
//'hist(r0)
//'@export
// [[Rcpp::export]]
arma::vec rand_left_trucnorm0(int n, double lower, double ratio=1.0){
	arma::vec y = arma::zeros<arma::vec>(n);
	int m = 0;
	if(lower<=0){
		arma::vec z = arma::randn<arma::vec>(ceil(ratio*(n - m)));
		arma::uvec idx = arma::find(z>lower);
		m += idx.n_elem;
		if(idx.n_elem>0){
		if(m<=n)
			y.subvec(0,m-1) = z.elem(idx);
		else
			y = z.elem(idx.subvec(0,n-1));
		}
		while(m < n){
			z = arma::randn<arma::vec>(ceil(ratio*(n - m)));
			idx = arma::find(z>lower);
			if(idx.n_elem>0){
				if(m + idx.n_elem <= n)
					y.subvec(m, m + idx.n_elem - 1) = z.elem(idx);
				else
					y.subvec(m,n-1) = z.elem(idx.subvec(0,n-m-1));
				m += idx.n_elem;
			}
		}
	} else{
		double alpha_star = 0.5*(lower+sqrt(lower*lower+4.0));
		arma::vec z = lower - log(arma::randu<arma::vec>(ceil(ratio*n)))/alpha_star;
    arma::vec log_rho_z = (z - alpha_star);
    log_rho_z %= -0.5*log_rho_z;
    arma::vec u = arma::randu<arma::vec>(ceil(ratio*n));
    arma::uvec idx = arma::find(log(u) < log_rho_z);
    m += idx.n_elem;
    if(idx.n_elem>0){
    if(m<=n)
    	y.subvec(0,m-1) = z.elem(idx);
    else
    	y = z.elem(idx.subvec(0,n-1));
    }
    while(m < n){
    	z = lower - log(arma::randu<arma::vec>(ceil(ratio*(n - m))))/alpha_star;
    	log_rho_z = (z - alpha_star);
    	log_rho_z %= -0.5*log_rho_z;
    	u = arma::randu<arma::vec>(ceil(ratio*(n - m)));
    	idx = arma::find(log(u) < log_rho_z);
    	if(idx.n_elem>0){
    		if(m + idx.n_elem <= n)
    			y.subvec(m, m + idx.n_elem - 1) = z.elem(idx);
    		else
    			y.subvec(m,n-1) = z.elem(idx.subvec(0,n-m-1));
    		m += idx.n_elem;
    	}
    }
	}
	return y;
}


//'@title Simulate left truncated normal distribution
//'@param n sample size
//'@param mu the location parameter
//'@param sigma the scale parameter
//'@param lower the lower bound
//'@return a vector of random numbers from a left truncated
//'normal distribution
//'@author Jian Kang <jiankang@umich.edu>
//'@examples
//'r <- rand_left_trucnorm(1000, mu=100, sigma=0.1, lower=200)
//'hist(r)
//'@export
// [[Rcpp::export]]
arma::vec rand_left_trucnorm(int n, double mu, double sigma,
                             double lower, double ratio=1.0){
	double lower0 = (lower - mu)/sigma;
	arma::vec y = rand_left_trucnorm0(n,lower0,ratio);
	y *= sigma;
	y += mu;
	return y;
}

//'@title Simulate right truncated normal distribution
//'@param n sample size
//'@param mu the location parameter
//'@param sigma the scale parameter
//'@param upper the upper bound
//'@return a vector of random numbers from a right truncated
//'normal distribution
//'@author Jian Kang <jiankang@umich.edu>
//'@examples
//'r <- rand_right_trucnorm(1000, mu=100, sigma=0.1, upper=90)
//'hist(r)
//'@export
// [[Rcpp::export]]
arma::vec rand_right_trucnorm(int n, double mu, double sigma,
                             double upper, double ratio=1.0){
	arma::vec y = rand_left_trucnorm(n,-mu,sigma,-upper,ratio);
	return -y;
}


void hs_one_step_update_big_p(arma::vec& betacoef, arma::vec& lambda,
                              double& sigma2_eps, double& tau2,
                           double& b_tau, arma::vec& b_lambda, arma::vec& mu, arma::vec& ys,  arma::mat& V, arma::vec& d,arma::vec& d2,
                           arma::vec& y, arma::mat& X, arma::mat& VD,
                           double& A2, double& A2_lambda,
                           double a_sigma, double b_sigma,
                           int p, int n){

	arma::vec lambda2 = lambda%lambda;
	double sigma_eps = sqrt(sigma2_eps);
	double tau = sqrt(tau2);
	double inv_tau2 = 1.0/tau2;
	arma::vec alpha_1 = arma::randn<arma::vec>(p)%lambda*sigma_eps*tau;
	arma::vec alpha_2 = arma::randn<arma::vec>(n)*sigma_eps;
	arma::mat LambdaVD = VD;
	for(int i=0;i<n;i++){
		LambdaVD.col(i) %= lambda;
	}
	arma::mat Z = arma::eye(n,n);
	Z += tau2*LambdaVD.t()*LambdaVD;
	arma::vec beta_s = arma::solve(Z,ys - VD.t()*alpha_1 - alpha_2,arma::solve_opts::fast);
	//std::cout << beta_s.subvec(0,1) << std::endl;
	betacoef = alpha_1 + tau2*lambda2%(VD*beta_s);
	//update lambda
	arma::vec betacoef2 = betacoef%betacoef;
	arma::vec inv_lambda2 = arma::randg<arma::vec>(p,distr_param(1.0,1.0));
	inv_lambda2 /= b_lambda + 0.5*betacoef2/tau2/sigma2_eps;
	b_lambda = randg<arma::vec>(p,distr_param(1.0, 1.0));
	b_lambda /= 1.0/A2_lambda+inv_lambda2;
	lambda = sqrt(1.0/inv_lambda2);
	//update tau2, sigma2_eps, b_tau and b_lambda
	mu = X*betacoef;
	arma::vec eps = y - mu;
	double sum_eps2 = arma::accu(eps%eps);
	double sum_beta2_inv_lambda2 = arma::accu(betacoef2%inv_lambda2);
	//double sum_inv_lambda2 = arma::accu(inv_lambda2);
	inv_tau2 = randg<double>(distr_param((1.0+p)/2.0,1.0/(b_tau+0.5*sum_beta2_inv_lambda2/sigma2_eps)));
	b_tau = randg<double>(distr_param(1.0,1.0/(1.0/A2 + inv_tau2)));
	tau2 = 1.0/inv_tau2;
	double inv_sigma2_eps = arma::randg<double>(distr_param(a_sigma+(p+n)/2, 1.0/(b_sigma+0.5*sum_beta2_inv_lambda2*inv_tau2+0.5*sum_eps2)));
	sigma2_eps = 1.0/inv_sigma2_eps;
}


void hs_one_step_update_big_n(arma::vec& betacoef,
                              arma::vec& lambda,
                              double& sigma2_eps, double& tau2, arma::vec& b_lambda,
                           double& b_tau, arma::vec& mu, arma::vec& ys,
                           arma::mat& V, arma::vec& d,arma::vec& d2,
                           arma::vec& y, arma::mat& X,
                           arma::vec& Xty, arma::mat& XtX_inv,
                           double A2, double A2_lambda,
                           double a_sigma, double b_sigma,
                           int p, int n){

	double inv_tau2 = 1.0/tau2;
	double tau = sqrt(tau2);
	double sigma_eps = sqrt(sigma2_eps);
	arma::vec taulambda= tau*lambda;
	arma::vec tau2lambda2 = taulambda%taulambda;
	arma::vec alpha_1 = arma::randn<arma::vec>(p)/d*sigma_eps;
	arma::vec alpha_2 = arma::randn<arma::vec>(p)%taulambda*sigma_eps;
	arma::vec ts = tau2lambda2%Xty;
	arma::vec Valpha_1 = V*alpha_1;
	arma::mat Z = XtX_inv;
	Z.diag() += tau2lambda2;
	arma::vec alpha = arma::solve(Z,ts - Valpha_1 - alpha_2)/sigma2_eps;
	betacoef = Valpha_1 + sigma2_eps*XtX_inv*alpha;
	arma::vec betacoef2 = betacoef%betacoef;
	arma::vec inv_lambda2 = arma::randg<arma::vec>(p,distr_param(1.0,1.0));
  inv_lambda2 /= b_lambda + 0.5*betacoef2/tau2/sigma2_eps;
  lambda = sqrt(1.0/inv_lambda2);
  b_lambda = randg<arma::vec>(p,distr_param(1.0, 1.0));
  b_lambda /= 1.0/A2_lambda+inv_lambda2;
  mu = X*betacoef;
	arma::vec eps = y - mu;
	double sum_eps2 = arma::accu(eps%eps);
	double sum_beta2_inv_lambda2 = arma::accu(betacoef%betacoef%inv_lambda2);
	//double sum_inv_lambda2 = arma::accu(inv_lambda2);
	inv_tau2 = randg<double>(distr_param((1.0+p)/2.0,1.0/(b_tau+0.5*sum_beta2_inv_lambda2/sigma2_eps)));
	b_tau = randg<double>(distr_param(1.0,1.0/(1.0/A2 + inv_tau2)));
	tau2 = 1.0/inv_tau2;
	double inv_sigma2_eps = arma::randg<double>(distr_param(a_sigma+(p+n)/2, 1.0/(b_sigma+0.5*sum_beta2_inv_lambda2*inv_tau2+0.5*sum_eps2)));
	sigma2_eps = 1.0/inv_sigma2_eps;
}

//'@title Fast Bayesian linear regression with horseshoe priors
//'@param y vector of n outcome variables
//'@param X n x p matrix of candidate predictors
//'@param mcmc_sample number of MCMC iterations saved
//'@param burnin number of iterations before start to save
//'@param thinning number of iterations to skip between two saved iterations
//'@param a_sigma shape parameter in the inverse gamma prior of the noise variance
//'@param b_sigma rate parameter in the inverse gamma prior of the noise variance
//'@param A_tau scale parameter in the half Cauchy prior of the global shrinkage parameter
//'@param A_lambda scale parameter in the half Cauchy prior of the local shrinkage parameter
//'@return a list object consisting of two components
//'\describe{
//'\item{post_mean}{a list object of four components for posterior mean statistics}
//'\describe{
//'\item{mu}{a vector of posterior predictive mean of the n training sample}
//'\item{betacoef}{a vector of posterior mean of p regression coeficients}
//'\item{lambda}{a vector of posterior mean of p local shrinkage parameters}
//'\item{sigma2_eps}{posterior mean of the noise variance}
//'\item{b_lambda}{posterior mean of the rate parameter in the prior for local shrinkage parameters}
//'\item{tau2}{posterior mean of the global parameter}
//'}
//'\item{mcmc}{a list object of three components for MCMC samples}
//'\describe{
//'\item{betacoef}{a matrix of MCMC samples of p regression coeficients}
//'\item{lambda}{a matrix of MCMC samples of p local shrinkage parameters}
//'\item{sigma2_eps}{a vector of MCMC samples of the noise variance}
//'\item{b_lambda}{a vector of MCMC samples of the rate parameter in the prior for local shrinkage parameters}
//'\item{tau2}{a vector of MCMC samples of the global shrinkage parameter}
//'}
//'}
//'@author Jian Kang <jiankang@umich.edu>
//'@examples
//'set.seed(2022)
//'dat1 <- sim_linear_reg(n=2000,p=200,X_cor=0.9,q=6)
//'res1 <- with(dat1,fast_horseshoe_lm(y,X))
//'dat2 <- sim_linear_reg(n=200,p=2000,X_cor=0.9,q=6)
//'res2 <- with(dat2,fast_horseshoe_lm(y,X))
//'tab <- data.frame(rbind(comp_sparse_SSE(dat1$betacoef,res1$post_mean$betacoef),
//'comp_sparse_SSE(dat2$betacoef,res2$post_mean$betacoef)),
//'time=c(res1$elapsed,res2$elapsed))
//'rownames(tab)<-c("n = 2000, p = 200","n = 200, p = 2000")
//'fast_horseshoe_tab <- tab
//'print(fast_horseshoe_tab)
//'@export
// [[Rcpp::export]]
Rcpp::List fast_horseshoe_lm(arma::vec& y, arma::mat& X,
                         int mcmc_sample = 500,
                         int burnin = 500, int thinning = 1,
                         double a_sigma = 0.0, double b_sigma = 0.0,
                         double A_tau = 1, double A_lambda = 1){

	arma::wall_clock timer;
	timer.tic();
	arma::vec d;
	arma::mat U;
	arma::mat V;
	arma::svd_econ(U,d,V,X);

	int p = X.n_cols;
	int n = X.n_rows;
	double sigma2_eps = 1;
	if(a_sigma!=0.0){
		sigma2_eps = b_sigma/a_sigma;
	}
	double A2 = A_tau*A_tau;
	double A2_lambda = A_lambda*A_lambda;
	double b_tau = 1;

	double tau2 = 1;
	arma::vec d2 = d%d;
	arma::vec ys = U.t()*y;



	arma::vec betacoef;
	arma::vec lambda;
	arma::vec b_lambda;
	lambda.ones(p);
	b_lambda.ones(p);
	arma::vec mu;

	arma::mat betacoef_list;
	arma::mat lambda_list;
	arma::vec sigma2_eps_list;
	arma::vec tau2_list;

	betacoef_list.zeros(p,mcmc_sample);
	lambda_list.zeros(p,mcmc_sample);
	sigma2_eps_list.zeros(mcmc_sample);
	tau2_list.zeros(mcmc_sample);


	if(p<n){

		arma::mat XtX_inv = V*diagmat(1.0/d2)*V.t();
		arma::vec Xty = X.t()*y;
		for(int iter=0;iter<burnin;iter++){
			hs_one_step_update_big_n(betacoef,lambda, sigma2_eps, tau2,b_lambda,
                         b_tau, mu, ys,  V,  d, d2, y, X, Xty,  XtX_inv,
                         A2, A2_lambda, a_sigma,  b_sigma, p,  n);
		}
		for(int iter=0;iter<mcmc_sample;iter++){
			for(int j=0;j<thinning;j++){
				hs_one_step_update_big_n(betacoef,lambda, sigma2_eps, tau2,b_lambda,
                             b_tau, mu, ys,  V,  d, d2, y, X, Xty,  XtX_inv,
                             A2, A2_lambda, a_sigma,  b_sigma, p,  n);
			}
			betacoef_list.col(iter) = betacoef;
			lambda_list.col(iter) = lambda;
			sigma2_eps_list(iter) = sigma2_eps;
			tau2_list(iter) = tau2;
		}
	} else{
		arma::mat VD = V;
		for(int j=0;j<p;j++){
			VD.row(j) %= d.t();
		}
		for(int iter=0;iter<burnin;iter++){
			hs_one_step_update_big_p(betacoef, lambda, sigma2_eps, tau2,
                         b_tau, b_lambda, mu, ys,  V,  d, d2, y,  X, VD,
                         A2, A2_lambda, a_sigma,  b_sigma, p,  n);
		}
		for(int iter=0;iter<mcmc_sample;iter++){
			for(int j=0;j<thinning;j++){
				hs_one_step_update_big_p(betacoef, lambda, sigma2_eps, tau2,
                          b_tau,b_lambda, mu, ys,  V,  d, d2, y,  X, VD,
                          A2, A2_lambda, a_sigma,  b_sigma, p,  n);
			}
			betacoef_list.col(iter) = betacoef;
			lambda_list.col(iter) = lambda;
			sigma2_eps_list(iter) = sigma2_eps;
			tau2_list(iter) = tau2;
		}

	}

	betacoef = arma::mean(betacoef_list,1);
	lambda = arma::mean(lambda_list,1);
	sigma2_eps = arma::mean(sigma2_eps_list);
	tau2 = arma::mean(tau2_list);

	Rcpp::List post_mean = Rcpp::List::create(Named("mu") = X*betacoef,
                                           Named("betacoef") = betacoef,
                                           Named("lambda") = lambda,
                                           Named("sigma2_eps") = sigma2_eps,
                                           Named("tau2") = tau2);
	Rcpp::List mcmc = Rcpp::List::create(Named("betacoef") = betacoef_list,
                                      Named("lambda") = lambda_list,
                                      Named("sigma2_eps") = sigma2_eps_list,
                                      Named("tau2") = tau2_list);

	double elapsed = timer.toc();
	return Rcpp::List::create(Named("post_mean") = post_mean,
                           Named("mcmc") = mcmc,
                           Named("elapsed") = elapsed);
}


void scalar_img_one_step_update(arma::vec& theta, arma::uvec& delta, arma::vec& lambda,
                              double& sigma2_eps, double& tau2,
                              double& b_tau, arma::vec& b_lambda,  arma::vec& betacoef,
                              arma::vec& y, arma::mat& X, arma::mat& Z, arma::mat& Phi, arma::vec& eps,
                              double& A2, double& A2_lambda,
                              double a_sigma, double b_sigma,
                              int p, int n, int L){


	//update delta
	for(int j=0;j<p;j++){
		//update mu_j
		arma::vec xphi_j = X.col(j)*betacoef(j);
		arma::vec eps_j = eps;
		if(delta(j)==1L){
			eps_j -= xphi_j;
		}
		arma::vec eps_j_1 = eps_j - xphi_j;
		double log_prob_1 = -0.5*arma::accu(eps_j_1%eps_j_1)/sigma2_eps;
		double log_prob_0 = -0.5*arma::accu(eps_j%eps_j)/sigma2_eps;
		double prob = 0.0;
		if(log_prob_1>log_prob_0){
			prob = 1.0/(1.0+exp(log_prob_0-log_prob_1));
		} else{
			prob = exp(log_prob_1-log_prob_0);
			prob = prob/(1+prob);
		}
		if(arma::randu<double>() < prob){
			delta(j) = 1L;
			eps = eps_j - xphi_j;
		} else{
			delta(j) = 0L;
			eps = eps_j;
		}
	}
	//update Z
	arma::uvec active_idx = arma::find(delta==1L);
	Z = X.cols(active_idx)*Phi.rows(active_idx);

	//update theta
	arma::vec lambda2 = lambda%lambda;
	double sigma_eps = sqrt(sigma2_eps);
	double tau = sqrt(tau2);
	double inv_tau2 = 1.0/tau2;
	arma::vec alpha_1 = arma::randn<arma::vec>(p)%lambda;
	alpha_1 *= sigma_eps*tau;
	arma::vec alpha_2 = arma::randn<arma::vec>(n);
	alpha_2 *= sigma_eps;
	arma::mat LambdaZt = Z.t();
	for(int i=0;i<n;i++){
		LambdaZt.col(i) %= lambda;
	}
	arma::mat ZZt = arma::eye(n,n);
	ZZt += tau2*LambdaZt.t()*LambdaZt;
	arma::vec theta_s = arma::solve(ZZt, y - Z*alpha_1 - alpha_2,arma::solve_opts::fast);
	theta = alpha_1 + tau2*lambda2%(Z.t()*theta_s);
	//update betacoef
	betacoef = Phi*theta;
	//update eps
	eps = y - Z*theta;


	//update lambda
	arma::vec theta2 = theta%theta;
	arma::vec inv_lambda2 = arma::randg<arma::vec>(p,distr_param(1.0,1.0));
	inv_lambda2 /= b_lambda + 0.5*theta2/tau2/sigma2_eps;
	b_lambda = randg<arma::vec>(p,distr_param(1.0, 1.0));
	b_lambda /= 1.0/A2_lambda+inv_lambda2;
	lambda = sqrt(1.0/inv_lambda2);

	//update tau2, sigma2_eps, b_tau and b_lambda
	double sum_eps2 = arma::accu(eps%eps);
	double sum_theta2_inv_lambda2 = arma::accu(theta2%inv_lambda2);
	inv_tau2 = randg<double>(distr_param((1.0+L)/2.0,1.0/(b_tau+0.5*sum_theta2_inv_lambda2/sigma2_eps)));
	b_tau = randg<double>(distr_param(1.0,1.0/(1.0/A2 + inv_tau2)));
	tau2 = 1.0/inv_tau2;
	double inv_sigma2_eps = arma::randg<double>(distr_param(a_sigma+(L+n)/2, 1.0/(b_sigma+0.5*sum_theta2_inv_lambda2*inv_tau2+0.5*sum_eps2)));
	sigma2_eps = 1.0/inv_sigma2_eps;
}

/*
//'@title Fast Bayesian Scalar-on-Image linear regression with Gaussian process priors
//'@param y vector of n outcome variables
//'@param X n x p matrix of candidate predictors
//'@param Phi p x L matrix of basis function
//'@param mcmc_sample number of MCMC iterations saved
//'@param burnin number of iterations before start to save
//'@param thinning number of iterations to skip between two saved iterations
//'@param a_sigma shape parameter in the inverse gamma prior of the noise variance
//'@param b_sigma rate parameter in the inverse gamma prior of the noise variance
//'@param A_tau scale parameter in the half Cauchy prior of the global shrinkage parameter
//'@param A_lambda scale parameter in the half Cauchy prior of the local shrinkage parameter
//'@return a list object consisting of two components
//'\describe{
//'\item{post_mean}{a list object of four components for posterior mean statistics}
//'\describe{
//'\item{mu}{a vector of posterior predictive mean of the n training sample}
//'\item{betacoef}{a vector of posterior mean of p regression coeficients}
//'\item{lambda}{a vector of posterior mean of p local shrinkage parameters}
//'\item{sigma2_eps}{posterior mean of the noise variance}
//'\item{tau2}{posterior mean of the global parameter}
//'}
//'\item{mcmc}{a list object of three components for MCMC samples}
//'\describe{
//'\item{betacoef}{a matrix of MCMC samples of p regression coeficients}
//'\item{lambda}{a matrix of MCMC samples of p local shrinkage parameters}
//'\item{sigma2_eps}{a vector of MCMC samples of the noise variance}
//'\item{tau2}{a vector of MCMC samples of the global shrinkage parameter}
//'}
//'}
//'@author Jian Kang <jiankang@umich.edu>
//'@examples
//'set.seed(2022)
//'dat1 <- sim_linear_reg(n=2000,p=200,X_cor=0.9,q=6)
//'res1 <- with(dat1,fast_horseshoe_lm(y,X))
//'dat2 <- sim_linear_reg(n=200,p=2000,X_cor=0.9,q=6)
//'res2 <- with(dat2,fast_horseshoe_lm(y,X))
//'tab <- data.frame(rbind(comp_sparse_SSE(dat1$betacoef,res1$post_mean$betacoef),
//'comp_sparse_SSE(dat2$betacoef,res2$post_mean$betacoef)),
//'time=c(res1$elapsed,res2$elapsed))
//'rownames(tab)<-c("n = 2000, p = 200","n = 200, p = 2000")
//'fast_horseshoe_tab <- tab
//'print(fast_horseshoe_tab)
//'@export
// [[Rcpp::export]]
Rcpp::List fast_scalar_img_lm(arma::vec& y_, arma::mat& X_, arma::mat>& Phi,
                             int mcmc_sample = 500,
                             int burnin = 500, int thinning = 1,
                             double a_sigma = 0.0, double b_sigma = 0.0,
                             double A_tau = 1, double A_lambda = 1){

	arma::wall_clock timer;
	timer.tic();

	arma::vec y;
	arma::mat X;

	if(p<n){
		arma::vec d;
		arma::mat U;
		arma::mat V;
		arma::svd_econ(U,d,V,X_);
		y = U.t()*y_;
		X = U.t()*X_;
	} else{
		y = y_;
		X = X_;
	}

	int p = X.n_cols;
	int n = X.n_rows;
	int L = Phi.n_cols;

	double sigma2_eps = 1;
	if(a_sigma!=0.0){
		sigma2_eps = b_sigma/a_sigma;
	}
	double A2 = A_tau*A_tau;
	double A2_lambda = A_lambda*A_lambda;
	double b_tau = 1;

	double tau2 = 1;
	arma::vec d2 = d%d;


	arma::vec theta;
	arma::uvec delta;
	arma::vec betacoef;
	arma::vec lambda;
	arma::vec b_lambda;
	arma::vec eps;
	arma::mat Z = X*Phi;

	theta.zeros(L);
	lambda.ones(L);
	b_lambda.ones(L);
	delta.ones(p);
	betacoef.zeros(p);
	eps.zeros(n);


	arma::mat theta_list;
	arma::umat delta_list;
	arma::mat betacoef_list;
	arma::mat lambda_list;
	arma::vec sigma2_eps_list;
	arma::vec tau2_list;


	delta_list.zeros(p,mcmc_sample);
	betacoef_list.zeros(p,mcmc_sample);
	theta_list.zeros(L,mcmc_sample);
	lambda_list.zeros(L,mcmc_sample);
	sigma2_eps_list.zeros(mcmc_sample);
	tau2_list.zeros(mcmc_sample);


		arma::mat XtX_inv = V*diagmat(1.0/d2)*V.t();
		arma::vec Xty = X.t()*y;
		for(int iter=0;iter<burnin;iter++){
			scalar_img_one_step_update(theta, delta, lambda,
                              sigma2_eps, tau2,
                              b_tau,  b_lambda,  betacoef,
                              y,  X,  Z,  Phi, eps, A2,  A2_lambda,
                              a_sigma, b_sigma, p, n, L);
		}
		for(int iter=0;iter<mcmc_sample;iter++){
			for(int j=0;j<thinning;j++){
				scalar_img_one_step_update(theta, delta, lambda,
                               sigma2_eps, tau2,
                                b_tau,  b_lambda,  betacoef,
                                y,  X,  Z,  Phi, eps, A2,  A2_lambda,
                                a_sigma, b_sigma, p, n, L);
			}
			theta_list.col(iter) = theta;
			delta_list.col(iter) = delta;
			betacoef_list.col(iter) = betacoef;
			lambda_list.col(iter) = lambda;
			sigma2_eps_list(iter) = sigma2_eps;
			tau2_list(iter) = tau2;
		}


	theta = arma::mean(theta_list,1);
	arma::vec delta_prob = arma::mean(delta_list*1.0,1);
	betacoef = arma::mean(betacoef_list,1);
	lambda = arma::mean(lambda_list,1);
	sigma2_eps = arma::mean(sigma2_eps_list);
	tau2 = arma::mean(tau2_list);

	Rcpp::List post_mean = Rcpp::List::create(Named("mu") = X_*betacoef,
                                           Named("betacoef") = betacoef,
                                           Named("theta") = theta,
                                           Named("delta_prob") = delta_prob,
                                           Named("lambda") = lambda,
                                           Named("sigma2_eps") = sigma2_eps,
                                           Named("tau2") = tau2);
	Rcpp::List mcmc = Rcpp::List::create(Named("betacoef") = betacoef_list,
                                      Named("lambda") = lambda_list,
                                      Named("sigma2_eps") = sigma2_eps_list,
                                      Named("tau2") = tau2_list);

	double elapsed = timer.toc();
	return Rcpp::List::create(Named("post_mean") = post_mean,
                           Named("mcmc") = mcmc,
                           Named("elapsed") = elapsed);
}*/
