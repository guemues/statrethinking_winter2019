data {
  
  int N; // Number of individuals
  int K; // Number of careers 

  int<lower=1,upper=K> response[N];
}

parameters {
  ordered[(K - 1)] q;
}

model{
  vector[K] p;
  
  
  p[1] = inv_logit(q[1]);
  
  for (i in 2:(K - 1) ) {
    p[i] = inv_logit(q[i]) - inv_logit(q[i - 1]);
  }
  
  p[K] = 1 - inv_logit(q[(K - 1)]);

  
  for (i in 1:N) {
    response[i] ~ categorical(p);
  }

}

