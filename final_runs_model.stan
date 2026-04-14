data {
  int<lower=0> N; // number of observations
  int<lower=1> K; // number of basis functions (sum across oppositions)
  int<lower=1> n_player; //no of players in dataset
  int<lower=1, upper = n_player> players[N]; //Indexes each unique player
  
  // Response variable
  int<lower=0> runs[N];
  
  // Predictor matrix for spline fit (year,opp)
  matrix[N, K] X;
  
  int<lower=0, upper=1> NO[N]; // censoring indicator variable
  
  int<lower=0,upper=1> inn_2[N]; //indicator for innings 2 (1 when it is)
  int<lower=0,upper=1> home[N]; //indicator for home (1 when at home)

  // Opposition information
  int<lower=1> n_oppositions; // number of oppositions in dataset
  int<lower=1, upper=n_oppositions> opp_id[N]; // opposition ID for each observation
  
  vector[N] age; //centre in R
}

transformed data {
  int N_uncensored = 0;
  int N_censored = 0;
  
  // Create counts of censored and non-censored observations
  for (i in 1:N) {
    if (NO[i] == 0) {
      N_uncensored += 1;
    } else {
      N_censored += 1;
    }
  }
  
  int uncensored_indices[N_uncensored];
  int censored_indices[N_censored];
  
  // Record indices for censored and uncensored observations
  {
    int u_idx = 1;
    int c_idx = 1;
    for (i in 1:N) {
      if (NO[i] == 0) {
        uncensored_indices[u_idx] = i;
        u_idx += 1;
      } else {
        censored_indices[c_idx] = i;
        c_idx += 1;
      }
    }
  }
  
  // Extract sparse components from dense matrix X
  int nnz = csr_extract_u(X)[N+1] - 1;  // Get number of non-zero elements
  vector[nnz] csr_w = csr_extract_w(X);
  array[nnz] int csr_v = csr_extract_v(X);
  array[N+1] int csr_u = csr_extract_u(X);
}
  
  
  parameters {
  vector[n_player] alpha; // intercept (shared across oppositions)
  vector[K] beta; // spline coefficients (opposition-specific due to block diagonal X)
  real<lower=0> phi; // dispersion parameter
  real<lower=0, upper=1> omega; // zero-inflation probability parameter
  
  real delta_inns; //Parameter for the innings indicator variable
  real delta_home; //Parameter for home 
  
  vector[n_player] xi_1; //Peak age parameter
  vector<lower=0>[n_player] xi_2; //ROC parameter
}

transformed parameters {
  vector[N] mu; // expected value of runs
  vector[N] age_func; 
  
  age_func = -xi_2[players].*square(age-xi_1[players]);
  
  //mu fitting
  mu = exp(alpha[players] + 
  delta_inns*to_vector(inn_2) + 
  delta_home*to_vector(home) + 
  csr_matrix_times_vector(N, K, csr_w, csr_v, csr_u, beta)+
  age_func);
}

model {
  // Priors 
  alpha ~ normal(3.28, 0.5);
  beta ~ normal(0,1);
  phi ~ gamma(10,10);
  omega ~ beta(1,1);
  
  delta_inns ~ normal(0,0.1);
  delta_home ~ normal(0,0.1);
  
  xi_1 ~ normal(0,0.5);
  xi_2 ~ lognormal(-3,0.5);
  

  // Likelihood for uncensored observations
  for (k in 1:N_uncensored) {
    int i = uncensored_indices[k];
    if (runs[i] == 0) {
      target += log_sum_exp(bernoulli_lpmf(1 | omega),
                           bernoulli_lpmf(0 | omega) +
                           neg_binomial_2_lpmf(runs[i] | mu[i], phi)); 
    } else {
      target += bernoulli_lpmf(0 | omega) + 
                neg_binomial_2_lpmf(runs[i] | mu[i], phi); 
    }
  }
  
  // Likelihood for censored observations
  for (m in 1:N_censored) {
    int i = censored_indices[m];
    target += bernoulli_lpmf(0 | omega) + 
              neg_binomial_2_lccdf(runs[i] | mu[i], phi);
  }
}

generated quantities {
  vector[N] log_lik; // initialises log-likelihood vector for each observation
  vector[N] y_rep; // initialises replicated data vector from predictive posterior
  
  // Calculate log-likelihood for each observation
  for (i in 1:N) {
    if (NO[i] == 0) {  // Uncensored observation
      if (runs[i] == 0) {
        log_lik[i] = log_sum_exp(bernoulli_lpmf(1 | omega),
                                bernoulli_lpmf(0 | omega) +
                                neg_binomial_2_lpmf(runs[i] | mu[i], phi));
      } else {
        log_lik[i] = bernoulli_lpmf(0 | omega) + 
                     neg_binomial_2_lpmf(runs[i] | mu[i], phi); 
      }
    } else {  // Censored observation
      log_lik[i] = bernoulli_lpmf(0 | omega) + 
                   neg_binomial_2_lccdf(runs[i] | mu[i], phi);
    }
    
    // Generate replicated data
    if (bernoulli_rng(omega)) {
      y_rep[i] = 0;
    } else {
      y_rep[i] = neg_binomial_2_rng(mu[i], phi);
    }
  }
}
