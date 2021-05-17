data {
  int N; // Number of observations
  int K; // NUmber of districts
  
  int<lower=1, upper=2> Uid[N]; // urban type
  int<lower=1, upper=K> D[N]; // district id
  int C[N]; // Observation results
}

parameters { 
  real ddD;
  real ddU;
  corr_matrix[2] Rho;
  vector<lower=0>[2] sigma;
  vector[2] dDU[K];
}

transformed parameters {
}

model{
  vector[N] u;
  vector[2] mu;
  
  Rho ~ lkj_corr(2);
  sigma ~ exponential(1);
  ddD ~ normal(0, 1.5);
  ddU ~ normal(0, 1.5);
  
  
  dDU ~ multi_normal([ddD, ddU]', diag_matrix(sigma) * Rho * diag_matrix(sigma));
  for(i in 1:N){
    u[i] = inv_logit(dDU[D[i], Uid[i]]);
  }  
  C ~ binomial(1, u);
}

generated quantities {
  vector[N] u;
  vector[N] log_lik;
  
  for(i in 1:N){
    u[i] = inv_logit(dDU[D[i], Uid[i]]);
  }
  
  for (i in 1:N) {
    log_lik[i] = binomial_lpmf(C[i] | 1, u[i]);
  }
}