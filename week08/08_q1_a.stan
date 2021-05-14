data {
  int N; // Number of observations
    
  int<lower=0, upper=1> P[N]; // Pred predictor
  int D[N]; // Number of frogs in each pod
  int S[N]; // Observation results
}

parameters {
  real bP; // effect of being
  real intercept;
}

model{
  
  vector[N] u;
  
  intercept ~ normal(0, 1);
  
  //u = inv_logit(A + bP * to_vector(P) + bZ[Z]);
  u = inv_logit(rep_vector(intercept, N)); // uncentered version
  
  S ~ binomial(D, u);
}


generated quantities {
  vector[N] u;
  vector[N] log_lik;
  
  u = inv_logit(rep_vector(intercept, N)); // uncentered version
  
  
  for (i in 1:N) {
    log_lik[i] = binomial_lpmf(S[i] | D[i], u[i]);
  }
}