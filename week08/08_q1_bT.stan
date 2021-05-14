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
  real bP; // effect of being
  real intercept;
}

model{
  
  vector[N] u;
  
  bP ~ normal(0, 1);
  intercept ~ normal(0, 1);
  
  //u = inv_logit(A + bP * to_vector(P) + bZ[Z]);
  u = inv_logit(rep_vector(intercept, N) + bP * to_vector(P)); // uncentered version
  
  S ~ binomial(D, u);
}


generated quantities {
  int ST_sim[T];
  
  vector[T] u;
  vector[T] log_lik;
  
  u = inv_logit(rep_vector(intercept, T) + bP * to_vector(PT)); // uncentered version
  
  for (i in 1:T) {
    log_lik[i] = binomial_lpmf(ST[i] | DT[i], u[i]);
  }
  
  for(i in 1:T){
    ST_sim[i] = binomial_rng(DT[i], u[i]);
  }
}