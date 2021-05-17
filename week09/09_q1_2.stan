data {
  int N; // Number of observations
  int K; // NUmber of districts
  
  int<lower=0, upper=1> U[N];
  int<lower=1, upper=K> D[N]; // district id
  int C[N]; // Observation results
}

parameters { 
  real<lower=0> dD_sigma;
  real ddD;
  
  real<lower=0> dU_sigma;
  real ddU;
  
  vector[K] dD;
  vector[K] dU;
}

transformed parameters {
}

model{
  vector[N] u;
  
  dD_sigma ~ exponential(1);
  ddD ~ normal(0, 1.5);
  
  dU_sigma ~ exponential(1);
  ddU ~ normal(0, 1.5);
  
  dD ~ normal(ddD, dD_sigma);
  dU ~ normal(ddU, dU_sigma);
  
  u = inv_logit(dD[D] + dU[D] .* to_vector(U));
  
  C ~ binomial(1, u);
}

generated quantities {
  vector[N] u;
  vector[N] log_lik;
  
  u = inv_logit(dD[D] + dU[D] .* to_vector(U)); // uncentered version
  
  for (i in 1:N) {
    log_lik[i] = binomial_lpmf(C[i] | 1, u[i]);
  }
}