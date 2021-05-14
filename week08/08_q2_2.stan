data {
  int N; // Number of observations
  int K; // NUmber of districts
  
  int<lower=1, upper=K> D[N]; // district id
  int C[N]; // Observation results
}

parameters {
  vector[K] dD; // varying intercept mean
  real dD_sigma; // varying intercept standart deviation
  real dD_mean; // varying intercept mean
}

model{
  
  vector[N] u;
  
  dD_sigma ~ exponential(1);
  dD_mean ~ normal(0, 1.5);
  
  //A ~ normal(A_mean, A_sigma); // centered version
  dD ~ normal(0, 1); // uncentered version
  
  //u = inv_logit(A);
  u = inv_logit((dD_mean + dD * dD_sigma)[D]); // uncentered version
  
  C ~ bernoulli(u);
}

generated quantities {
  vector[N] u;
  vector[N] log_lik;
  
  u = inv_logit(dD_mean + dD * dD_sigma); // uncentered version
  
  
  for (i in 1:N) {
    log_lik[i] = bernoulli_lpmf(C[i] | u[D]);
  }
}