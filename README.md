# R package `fastBayesReg`


## Overview

fastBayesReg implements Bayesian linear regression with different priors. The current version includes the normal prior and the horseshoe prior. 


## Install from Github

Install package `fastBayesReg` from Github with 

```r
devtools::install_github("kangjian2016/fastBayesReg")
library(fastBayesReg)
```

## Examples: Normal Priors

```r

set.seed(2022)
dat1 <- sim_linear_reg(n=2000,p=200,X_cor=0.9,q=6)
res1 <- with(dat1,fast_normal_lm(y,X))
dat2 <- sim_linear_reg(n=200,p=2000,X_cor=0.9,q=6)
res2 <- with(dat2,fast_normal_lm(y,X))
tab <- data.frame(rbind(comp_sparse_SSE(dat1$betacoef,res1$post_mean$betacoef),
comp_sparse_SSE(dat2$betacoef,res2$post_mean$betacoef)),
time=c(res1$elapsed,res2$elapsed))
rownames(tab)<-c("n = 2000, p = 200","n = 200, p = 2000")
fast_normal_tab <- tab
print(fast_normal_tab)

```

## Examples: Horseshoe Priors

```r
set.seed(2022)
dat1 <- sim_linear_reg(n=2000,p=200,X_cor=0.9,q=6)
res1 <- with(dat1,fast_horseshoe_lm(y,X))
dat2 <- sim_linear_reg(n=200,p=2000,X_cor=0.9,q=6)
res2 <- with(dat2,fast_horseshoe_lm(y,X))
tab <- data.frame(rbind(comp_sparse_SSE(dat1$betacoef,res1$post_mean$betacoef),
comp_sparse_SSE(dat2$betacoef,res2$post_mean$betacoef)),
time=c(res1$elapsed,res2$elapsed))
rownames(tab)<-c("n = 2000, p = 200","n = 200, p = 2000")
fast_horseshoe_tab <- tab
print(fast_horseshoe_tab)
```

