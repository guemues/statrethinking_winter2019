data {
  int N; // Number of observations
  int K; // NUmber of districts
  
  int<lower=1, upper=K> D[N]; // district id
  int C[N]; // Observation results
}

parameters { // varying intercept standart deviation
  vector[K] dD; // varying intercept mean
}

model{
  vector[N] u;
  
  dD ~ normal(0, 1.5);
  
  //u = inv_logit(A);
  u = inv_logit(dD[D]); // uncentered version
  
  C ~ bernoulli(u);
}

generated quantities {
  vector[N] u;
  vector[N] log_lik;
  
  u = inv_logit(dD[D]); // uncentered version
  
  for (i in 1:N) {
    log_lik[i] = bernoulli_lpmf(C[i] | u[D]);
  }
}