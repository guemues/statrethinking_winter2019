data {
  
  int N; // Number of individuals
  int K; // Number of careers 
  
  vector[N] A;
  vector[N] C;
  vector[N] I;
  vector[N] IA;
  vector[N] IC;
  
  vector[N] F;
  int E[N];
  vector[N] AGE;
  
  int<lower=1,upper=K> response[N];
}

parameters {
  real bA;
  real bC;
  real bI;
  real bIA;
  real bIC;
  
  real bAGE;
  real bE;
  real bF;
  
  ordered[(K - 1)] q;
  simplex[7] delta_education;
}

model{
  vector[N] u;
  vector[8] education_effect;
  
  delta_education ~ dirichlet([2, 2, 2, 2, 2, 2, 2]');
  education_effect = cumulative_sum(append_row([0]', delta_education));
  
  bE ~ normal(0, 1);
  bF ~ normal(0, 1);
  
  bA ~ normal(0, 1);
  bC ~ normal(0, 1);
  bI ~ normal(0, 1);
  
  bIA ~ normal(0, 1);  
  bIC ~ normal(0, 1); 
  bAGE ~ normal(0, 1);
  
  u = bE * education_effect[E] + bA * A + bC * C + bI * I + bIA * IA + bIC * IC + bAGE * AGE + bF * F;
  
  response ~ ordered_logistic(u, q);
  
}



generated quantities {
  vector[N] u;
  vector[8] education_effect;
  vector[N] log_lik;
  
  education_effect = cumulative_sum(append_row([0]', delta_education));
  
  u = bE * education_effect[E] + bA * A + bC * C + bI * I + bIA * IA + bIC * IC + bAGE * AGE + bF * F;
  
  for (i in 1:N) {
    log_lik[i] = ordered_logistic_lpmf(response[i] | u[i], q);
  }
  
}