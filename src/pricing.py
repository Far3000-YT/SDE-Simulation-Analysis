import numpy as np
from scipy.stats import norm
from sde_simulator.simulators import simulate_gbm_em_vectorized, simulate_gbm_milstein_vectorized

def black_scholes_call(S0, K, r, sigma, T):
    #using black scholes model here to price an option !
    if sigma <= 0 or T <= 0: #edge cases
        #if vol or time is zero, payoff is deterministic
        payoff_at_expiry = np.maximum(S0 * np.exp(r * T) - K, 0)
        return np.exp(-r * T) * payoff_at_expiry
    
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    price = S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return price

def price_european_option_mc(S0, K, r, sigma, T, dt, num_paths, option_type = 'call', scheme = 'EM', Z = None):
    #pricing a European option using Monte Carlo simulation of GBM
    #S0, K, r, sigma, T : option and model parameters
    #dt : time step for simulation
    #PATHS : number of simulation paths
    #option_type : 'call' or 'put'
    #scheme : 'EM' or 'Milstein' for the simulation scheme we'll use
    #Z : pre-generated random numbers
    #returns the Estimated option price

    #simulation here, chosen in the function parameter
    if scheme.lower() == 'em':
        t, S_paths = simulate_gbm_em_vectorized(S0, r, sigma, T, dt, num_paths, Z=Z) 
    elif scheme.lower() == 'milstein':
        t, S_paths = simulate_gbm_milstein_vectorized(S0, r, sigma, T, dt, num_paths, Z=Z)
    else:
        raise ValueError("Unsupported scheme. Choose 'EM' or 'Milstein' !!!")

    #get the final prices
    S_T = S_paths[-1, :]

    #calculate payoffs
    if option_type.lower() == 'call':
        payoffs = np.maximum(S_T - K, 0)
    elif option_type.lower() == 'put':
        payoffs = np.maximum(K - S_T, 0)
    else:
        raise ValueError("Option type must be 'call' or 'put'")

    #discounted expected payoff (Monte Carlo estimate)
    mc_price = np.exp(-r * T) * np.mean(payoffs)

    return mc_price