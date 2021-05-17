library(rethinking)
library(dplyr)

a <- 3.5

b <- -1 

sigma_a <- 1
sigma_b <- .5

rho <- (-0.1) # 0.7

Mu <- c(a, b)

cov_ab <- sigma_a * sigma_b * rho

N_cafes <- 20

library(MASS)

sigmas <- c(sigma_a, sigma_b)

Sigma <- diag(sigmas) %*% matrix(c(1, rho, rho, 1), nrow= 2) %*% diag(sigmas)

set.seed(5)

vary_effects <- mvrnorm(N_cafes, Mu, Sigma)


## R code 13.8
a_cafe <- vary_effects[,1]
b_cafe <- vary_effects[,2]

## R code 13.9
plot( a_cafe , b_cafe , col=rangi2 ,
      xlab="intercepts (a_cafe)" , ylab="slopes (b_cafe)" )

# overlay population distribution
library(ellipse)
for ( l in c(0.1,0.3,0.5,0.8,0.99) )
  lines(ellipse(Sigma,centre=Mu,level=l),col=col.alpha("black",0.2))


N_visits <- 100
afternoon <- rep(0:1,N_visits*N_cafes/2)
cafe_id <- rep( 1:N_cafes , each=N_visits )
mu <- a_cafe[cafe_id] + b_cafe[cafe_id]*afternoon
sigma <- 0.5  # std dev within cafes
wait <- rnorm( N_visits*N_cafes , mu , sigma )
d <- data.frame( cafe=cafe_id , afternoon=afternoon , wait=wait )


dat_list_tutorial_01 <- list(
  ID_c=d$cafe,
  A=d$afternoon,
  y=d$wait,
  N_o=N_visits * N_cafes,
  N_c=N_cafes,
  N_v=N_visits
)

tutorial_00 <- stan(file='week09/09_tutorial_00.stan', data=dat_list_tutorial_01, cores=4)
tutorial_01 <- stan(file='week09/09_tutorial_01.stan', data=dat_list_tutorial_01, cores=4)

estimations <- matrix(c(as.matrix(tutorial_01, pars = c("bC_e")) %>% apply(2, mean) %>% as.numeric, as.matrix(tutorial_01, pars = c("bA_e")) %>% apply(2, mean) %>% as.numeric),ncol=2)
df <- data.frame(estimations, type='estimation', id=as.factor(1:20)) %>% rbind(data.frame(vary_effects, type='real', id=as.factor(1:20)))

loo::loo_compare(loo(tutorial_00), loo(tutorial_01))

a_real <- vary_effects[, 1]
b_real <- vary_effects[, 2]


a1 <- sapply( 1:N_cafes ,
              function(i) mean(wait[cafe_id==i & afternoon==0]) )
b1 <- sapply( 1:N_cafes ,
              function(i) mean(wait[cafe_id==i & afternoon==1]) ) - a1


a2 <- estimations[, 1]
b2 <- estimations[, 2]

# plot both and connect with lines
plot( a1 , b1 , xlab="intercept" , ylab="slope" ,
      pch=16 , col=rangi2 , ylim=c( min(b1)-0.1 , max(b1)+0.1 ) ,
      xlim=c( min(a1)-0.1 , max(a1)+0.1 ) )
points( a2 , b2 , pch=1 )
for ( i in 1:N_cafes ) lines( c(a1[i],a2[i]) , c(b1[i],b2[i]) )


sum(abs(a_real - a1))
sum(abs(a_real - a2))

sum(abs(b_real - b1))
sum(abs(b_real - b2))
