data {
  int N; // Number of observations
  int K; // NUmber of districts
  
  real A[N];
  int<lower=1, upper=4> Ch[N];
  int<lower=0, upper=1> U[N]; // is urban
  int<lower=1, upper=K> D[N]; // district id
  int C[N]; // Observation results
}

parameters { 
  real ddD;
  real ddU;
  real dA;
  
  simplex[4] delta_dCh;
  
  corr_matrix[2] Rho;
  vector<lower=0>[2] sigma;
  vector[2] dDU[K];
}

transformed parameters {
}

model{
  vector[N] u;
  vector[4] dCh;
  
  Rho ~ lkj_corr(2);
  sigma ~ exponential(1);
  ddD ~ normal(0, 1.5);
  ddU ~ normal(0, 1.5);
  
  delta_dCh ~ dirichlet([2, 2, 2, 2]');
  
  dA ~ normal(0, 1.5);
  dDU ~ multi_normal([ddD, ddU]', diag_matrix(sigma) * Rho * diag_matrix(sigma));
  
  dCh = cumulative_sum(append_row([0]', delta_dCh));
  u = inv_logit(to_vector(dDU[D][1:N, 1]) + to_vector(dDU[D][1:N, 2]) .* to_vector(U) + dA * to_vector(A)  + dCh[Ch] );
  
  C ~ binomial(1, u);
}

generated quantities {
  vector[N] u;
  vector[N] log_lik;
  vector[4] dCh;
  
  dCh = cumulative_sum(append_row([0]', delta_dCh));
  u = inv_logit(to_vector(dDU[D][1:N, 1]) + to_vector(dDU[D][1:N, 2]) .* to_vector(U) + dA * to_vector(A) + dCh[Ch] );
  
  for (i in 1:N) {
    log_lik[i] = binomial_lpmf(C[i] | 1, u[i]);
  }
}