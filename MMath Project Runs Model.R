library(rstan)
library(dplyr)
library(ggplot2)
library(bayesplot)
library(loo)
library(coda)
library(splines)
library(Matrix)

load('30ODIgames.Rdata')

#Data Prep####
prepare_data = function(df) {
  
  df_clean = distinct(df) #remove duplicate rows
  
  #Combine low obs teams
  other_teams = c("EAf", "JER", "ICC", "Afr", "Asia")
  df_clean$Opposition[df_clean$Opposition %in% other_teams] = "Other"
  
  #get innings as a binary in (0,1) instead of (1,2)
  df_clean$Inns = df_clean$Inns -1
  
  #Give new id's to players for subset
  df_clean$ID = as.numeric(as.factor(df_clean$ID))
  
  #standardise age
  df_clean$Age_std = (df_clean$Age - mean(df_clean$Age))/sd(df_clean$Age)
  
  #Fit the spline
  spline_fit = function(opp_df) {
    
    # Get actual year range for this opposition
    year_range = range(opp_df$Year)
    n_obs = nrow(opp_df)
    
    #Should be >=25 obs per spline basis 
    max_knots = floor(n_obs / 25) - 3
    n_knots = max(0, min(5, max_knots)) #8 knots max
    
    if(n_knots >= 1) {
      # Place knots at quantiles
      knot_positions = quantile(opp_df$Year, 
                                probs = seq(0, 1, length.out = n_knots + 2)[2:(n_knots+1)])
      B = bs(opp_df$Year,
             knots = knot_positions,
             degree = 3,
             Boundary.knots = year_range,  #Use actual year range
             intercept = FALSE)
    } else {
      #For teams who played too little
      B = bs(opp_df$Year,
             knots = NULL,
             degree = 3,
             Boundary.knots = year_range,
             intercept = FALSE)}
    
    return(matrix(B, nrow = nrow(B), ncol = ncol(B)))
  }
  
  #Split by opposition
  opp_names = split(df_clean, df_clean$Opposition)
  B_list = list()
  
  for(opp in names(opp_names)) {
    opp_df = opp_names[[opp]]
    B_list[[opp]] = spline_fit(opp_df)
  }
  
  # Create sparse block diagonal matrix from each opposition spline
  B_spline = bdiag(B_list)
  
  # Store column mapping
  n_basis_per_opp = sapply(B_list, ncol)
  opp_names_vec = names(opp_names)
  
  
  # Prepare data for Stan
  stan_data = list(
    N = nrow(df_clean),
    K = ncol(B_spline),
    runs = df_clean$Runs,
    X = as.matrix(B_spline),  # Convert to dense for Stan CSR conversion
    NO = df_clean$NO,
    n_oppositions = length(opp_names_vec),
    opp_id = as.numeric(factor(df_clean$Opposition)),
    inn_2 = df_clean$Inns,
    home = df_clean$Home,
    n_player = length(unique(df_clean$ID)),
    players = df_clean$ID,
    age = df_clean$Age_std
  )
  
  #Return for input into fit function
  return(list(
    stan_data = stan_data, 
    df_clean = df_clean, 
    B = B_spline,
    B_list = B_list,
    opp_names = opp_names_vec,
    n_basis_per_opp = n_basis_per_opp
  ))
}

#Fit Model####
fit_model = function(stan_data, iter = 2000, warmup = 1000, chains = 1, 
                     adapt_delta = 0.95, max_treedepth = 15) {
  # Create initial starting chain values
  init_list = list()
  mean_runs = mean(stan_data$runs)
  
  for (chain in 1:chains) {
    init_list[[chain]] = list(
      alpha = rep(rnorm(1, log(mean_runs), 0.5), stan_data$n_player),
      beta = rnorm(stan_data$K, 0, 0.5),
      xi_1 = rep(rnorm(1, 0, 0.5), stan_data$n_player),
      xi_2 = rep(rlnorm(1, -3, 0.5), stan_data$n_player),
      phi = max(0.01, rgamma(1, 10, 10)),
      omega = runif(1, 0, 0.1),
      delta_inns = rnorm(1, 0, 0.1),
      delta_home = rnorm(1, 0, 0.1)
    )
  }
  
  for (chain in 1:chains) {
    cat(sprintf("  Chain %d: alpha[1:3]=[%.3f,%.3f,%.3f], beta[1:3]=[%.3f,%.3f,%.3f], xi_1[1:3]=[%.3f,%.3f,%.3f], xi_2[1:3]=[%.3f,%.3f,%.3f]\n",
                chain,
                init_list[[chain]]$alpha[1], init_list[[chain]]$alpha[2], init_list[[chain]]$alpha[3],
                init_list[[chain]]$beta[1], init_list[[chain]]$beta[2], init_list[[chain]]$beta[3],
                init_list[[chain]]$xi_1[1], init_list[[chain]]$xi_1[2], init_list[[chain]]$xi_1[3],
                init_list[[chain]]$xi_2[1], init_list[[chain]]$xi_2[2], init_list[[chain]]$xi_2[3]))
    cat(sprintf("      phi=%.3f, omega=%.3f, delta_inns=%.3f, delta_home=%.3f\n",
                init_list[[chain]]$phi, 
                init_list[[chain]]$omega,
                init_list[[chain]]$delta_inns,
                init_list[[chain]]$delta_home))
  }
  
  fit = sampling(
    cached_model,
    data = stan_data,
    iter = iter,
    warmup = warmup,
    init = init_list,
    chains = chains,
    control = list(adapt_delta = adapt_delta, max_treedepth = max_treedepth),
    refresh = 100,
    pars = c("alpha", "beta", "xi_1", "xi_2", "phi", "omega", "mu", "delta_inns", "delta_home", "log_lik", "y_rep")
  )
  
  return(fit)
}

#model summary####
summary_report = function(results, n_alpha = 10) {
  
  cat("Number of observations:", nrow(results$df_clean), "\n")
  cat("Censored observations:", sum(results$df_clean$NO), "\n")
  cat("Number of basis functions:", results$stan_data$K, "\n")
  cat("Number of oppositions:", results$stan_data$n_oppositions, "\n")
  cat("Number of players:", results$stan_data$n_player, "\n\n")
  
  cat("\nOpposition Totals (with year ranges):\n")
  for(i in seq_along(results$opp_names)) {
    opp_data = results$df_clean[results$df_clean$Opposition == results$opp_names[i], ]
    opp_n = nrow(opp_data)
    year_range = range(opp_data$Year)
    cat(sprintf("  %s: %d observations (years %d-%d)\n", 
                results$opp_names[i], opp_n, year_range[1], year_range[2]))
  }
  
  # Extract parameters
  post = results$posterior_samples
  
  for (j in 1:n_alpha) {
    cat(sprintf("  alpha[%d]: %.3f ± %.3f\n", 
                j, 
                mean(post$alpha[, j]), 
                sd(post$alpha[, j])))
  }
  cat("Dispersion (phi):", round(mean(post$phi), 3),
      "±", round(sd(post$phi), 3), "\n")
  cat("Zero-inflation probability (omega):", round(mean(post$omega), 3), 
      "±", round(sd(post$omega), 3), "\n")
  cat("Innings effect (delta_inns):", round(mean(post$delta_inns), 3),
      "±", round(sd(post$delta_inns), 3), "\n")
  cat("Home effect (delta_home):", round(mean(post$delta_home), 3),
      "±", round(sd(post$delta_home), 3), "\n")
  
  # Spline coefficients by opposition
  cat("\nSpline Coefficients\n")
  
  cum_basis = cumsum(c(0, results$n_basis_per_opp))
  
  for (opp_idx in seq_along(results$opp_names)) {
    
    n_basis = results$n_basis_per_opp[opp_idx]
    start_idx = cum_basis[opp_idx] + 1
    end_idx = cum_basis[opp_idx + 1]
    
    # Get year range for this opposition
    opp_years = range(results$df_clean$Year[results$df_clean$Opposition == results$opp_names[opp_idx]])
    
    cat(sprintf("\n%s (%d basis functions, years %d-%d):\n", 
                results$opp_names[opp_idx], n_basis, opp_years[1], opp_years[2]))
    
    for (i in 1:n_basis) {
      col_idx = start_idx + i - 1
      cat(sprintf("  beta[%d]: %.3f ± %.3f\n", 
                  i, 
                  mean(post$beta[, col_idx]), 
                  sd(post$beta[, col_idx])))
    }
  }
}

main = function(df) {
  #apply prepare_data
  prepared_data = prepare_data(df)
  stan_data = prepared_data$stan_data
  df_clean = prepared_data$df_clean
  
  #fit the model and print results
  fit = fit_model(stan_data)
  results = list(
    fit = fit,
    diagnostics = diagnostics,
    df_clean = df_clean,
    stan_data = stan_data,
    B = prepared_data$B,
    B_list = prepared_data$B_list,
    opp_names = prepared_data$opp_names,
    n_basis_per_opp = prepared_data$n_basis_per_opp,
    model_summary = summary(fit),
    posterior_samples = extract(fit)
  )
  
  return(results)
}

#Run and save model
runs_results = main(df)
saveRDS(zinb_spline_results, 'runs_results.rds')

summary_report(runs_results, n_alpha = 10)
