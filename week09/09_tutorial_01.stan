
data {
  int<lower=0> N_o; // Number of obsercations
  int<lower=0> N_c; // Number of cafee
  int<lower=0> N_v; // Number of visits
  
  int<lower=1, upper=N_c> ID_c[N_o]; // Cafe id
  int<lower=0, upper=1> A[N_o]; // is afternoon
  real y[N_o]; // Observatios
}

parameters {
  real bbC;
  real bbA;
  real sigma_bC;
  real sigma_bA;
  
  real sigma;

  vector[N_c] bC; // Cafe intercept
  vector[N_c] bA; // Cafe afternoon effect
  
}

transformed parameters{
  vector[N_c] bC_e;
  vector[N_c] bA_e;
  
  bC_e =  bC ;
  bA_e = bA;
}

model {
  vector[N_o] u;
  
  sigma ~ exponential(1);
  
  sigma_bA ~ exponential(1);
  sigma_bC ~ exponential(1);
  
  bbA ~ normal(-2, 2);
  bbC ~ normal(5, 10);
  
  bC ~ normal(bbC, sigma_bC);
  bA ~ normal(bbA, sigma_bA);
  
  u = bC_e[ID_c] + bA_e[ID_c] .* to_vector(A) ;
  y ~ normal(u, sigma);
}


generated quantities {
  vector[N_o] u;
  vector[N_o] log_lik;
  
  u = bC_e[ID_c] + bA_e[ID_c] .* to_vector(A) ;
    
  for (i in 1:N_o) {
    log_lik[i] = normal_lpdf(y[i] | u[i], sigma);
  }
  
}

