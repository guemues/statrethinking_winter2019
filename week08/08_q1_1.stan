data {
  int N; // Number of observations
  
  int D[N]; // Number of frogs in each pod
  int S[N]; // Observation results
}

parameters {
  vector[N] A; // varying intercept
    real A_sigma; // varying intercept standart deviation
    real A_mean; // varying intercept mean
}

model{
  
  vector[N] u;
  
  A_sigma ~ exponential(1);
  A_mean ~ normal(0, 1.5);
  
  //A ~ normal(A_mean, A_sigma); // centered version
  A ~ normal(0, 1); // uncentered version
  
  //u = inv_logit(A);
  u = inv_logit(A_mean + A * A_sigma); // uncentered version
  
  S ~ binomial(D, u);
}

generated quantities {
  vector[N] u;
  vector[N] log_lik;
  
  u = inv_logit(A_mean + A * A_sigma); // uncentered version
  
  
  for (i in 1:N) {
    log_lik[i] = binomial_lpmf(S[i] | D[i], u[i]);
  }
}