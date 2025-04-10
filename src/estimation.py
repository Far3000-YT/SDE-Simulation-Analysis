import numpy as np
from scipy.stats import norm 


def ou_neg_log_likelihood(params, data, dt):
    #unpacks parameters
    theta, kappa, sigma = params
    n = len(data)
    
    #parameter constraints check (avoid math errors)
    #theta and sigma must be positive
    #also handle edge case where theta is extremely small (variance denominator)
    if theta <= 1e-6 or sigma <= 1e-6: 
        return np.inf #return infinity if parameters are invalid

    x_prev = data[:-1] #data points from X_0 to X_{N-1}
    x_curr = data[1:]  #data points from X_1 to X_N
    
    #calculate mean and variance of transitions based on current params
    term_exp = np.exp(-theta * dt)
    mean = x_prev * term_exp + kappa * (1.0 - term_exp)
    
    variance = (sigma**2 / (2.0 * theta)) * (1.0 - np.exp(-2.0 * theta * dt))
    
    #avoid variance being zero or negative
    if variance <= 1e-8: 
         return np.inf

    #calculate log-likelihood contributions using Normal PDF
    #scipy.stats.norm.logpdf gives log of probability density
    log_likelihood_contributions = norm.logpdf(x_curr, loc=mean, scale=np.sqrt(variance))
    
    #sum contributions for total log-likelihood
    log_likelihood = np.sum(log_likelihood_contributions)
    
    #return NEGATIVE log-likelihood for minimization
    #add checks for NaN or Inf in likelihood (can happen with bad params)
    if np.isnan(log_likelihood) or np.isinf(log_likelihood):
        return np.inf 
        
    return -log_likelihood



#new approximation, since we don't find correct values with the firts function

def ou_neg_log_likelihood_approx(params, data, dt):
    #unpacks parameters
    theta, kappa, sigma = params
    n = len(data)
    
    #parameter constraints check
    if theta <= 1e-6 or sigma <= 1e-6: 
        return np.inf 

    x_prev = data[:-1] 
    x_curr = data[1:]  
    
    #calculate APPROXIMATE mean and variance based on EM step
    mean_approx = x_prev + theta * (kappa - x_prev) * dt
    variance_approx = sigma**2 * dt
    
    #avoid variance being zero or negative
    if variance_approx <= 1e-9: # Adjusted tolerance slightly
         return np.inf

    #calculate log-likelihood contributions using Normal PDF
    log_likelihood_contributions = norm.logpdf(x_curr, loc=mean_approx, scale=np.sqrt(variance_approx))
    
    #sum contributions
    log_likelihood = np.sum(log_likelihood_contributions)
    
    #return NEGATIVE log-likelihood
    if np.isnan(log_likelihood) or np.isinf(log_likelihood):
        return np.inf 
        
    return -log_likelihood