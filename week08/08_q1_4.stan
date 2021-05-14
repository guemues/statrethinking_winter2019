data {
  int N; // Number of observations
  
  int<lower=1, upper=2> Z[N]; // Size predictor  
  int<lower=0, upper=1> P[N]; // Pred predictor
  int D[N]; // Number of frogs in each pod
  int S[N]; // Observation results
}

parameters {
  vector[N] A; // varying intercept
  
  vector[2] bZ; // effect of being z
  real bP; // effect of being
  real A_sigma; // varying intercept standart deviation
  real A_mean; // varying intercept mean
}

model{
  
  vector[N] u;
  
  A_sigma ~ exponential(1);
  A_mean ~ normal(0, 1.5);
  
  bP ~ normal(-0.5 , 1);
  bZ ~ normal(0, 0.5);
  
  //A ~ normal(A_mean, A_sigma); // centered version
  A ~ normal(0, 1); // uncentered version
  
  //u = inv_logit(A + bP * to_vector(P) + bZ[Z]);
  u = inv_logit(A_mean + A * A_sigma + bP * to_vector(P) + bZ[Z]); // uncentered version
  
  S ~ binomial(D, u);
}


generated quantities {
  vector[N] u;
  vector[N] log_lik;
  
  u = inv_logit(A_mean + A * A_sigma + bP * to_vector(P) + bZ[Z]); // uncentered version
  
  
  for (i in 1:N) {
    log_lik[i] = binomial_lpmf(S[i] | D[i], u[i]);
  }
}

