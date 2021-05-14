data {
  int N; // Number of observations
  int T;
  
  int<lower=0, upper=1> P[N]; // Pred predictor
  int D[N]; // Number of frogs in each pod
  int S[N]; // Observation results
    
  int<lower=0, upper=1> PT[T]; // Pred predictor
  int DT[T]; // Number of frogs in each pod
  int ST[T]; // Observation results
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
  
  A ~ normal(0, 1); // uncentered version
  
  
  //u = inv_logit(A + bP * to_vector(P) + bZ[Z]);
  u = inv_logit(A_mean + A * A_sigma); // uncentered version
  
  S ~ binomial(D, u);
}


generated quantities {
  int ST_sim[T];

  vector[T] u;
  vector[T] log_lik;

  for (i in 1:T) {
    u[i] = inv_logit(normal_rng(A_mean, A_sigma));
  }

  for (i in 1:T) {
    log_lik[i] = binomial_lpmf(ST[i] | DT[i], u[i]);
  }

  for(i in 1:T){
    ST_sim[i] = binomial_rng(DT[i], u[i]);
  }
}