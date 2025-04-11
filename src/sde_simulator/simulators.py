import numpy as np

def simulate_gbm_em(s0, mu, sigma, T, dt):
    #Geometric Brownian Motion using Euler-Maruyama.
    #s0 = initial asset price
    #mu = drift coefficient
    #sigma = volatility coefficient
    #T = time horizon (in years)
    #dt = time step size

    #returned : t and S (array of time points then array of simulated asset prices at points, x and y in a graph basically)


    n_steps = int(T / dt)  #number of time steps
    t = np.linspace(0, T, n_steps + 1) #create time points between 0 and T, and steps (x axis basically)
    S = np.zeros(n_steps + 1)
    S[0] = s0 #we set the array for y (the price) now we set first value with function parameter of the initial price


    for i in range(n_steps):
        z = np.random.normal(0, 1) #basically number that tends to 0, deviation of 1

        #now we calculate the next price
        #S[i+1] ≈ S[i] + μ * S[i] * Δt + σ * S[i] * Z * sqrt(Δt)
        #'dt' is for our finite time step Δt
        
        current_price = S[i] #defining it since we'll use it few times
        drift_term = mu * current_price * dt 
        sqrt_dt = np.sqrt(dt) #pre calculating sqrt for efficiency, not necessary tho
        random_shock_term = sigma * current_price * z * sqrt_dt #calculation of the random shock term -> sigma * S(t) * Z * sqrt(dt)
        
        #next price calculation !
        S[i+1] = current_price + drift_term + random_shock_term

    #return the time points and the simulated path here
    return t, S


def simulate_gbm_em_vectorized(s0, mu, sigma, T, dt, num_paths, Z = None):
    n_steps = int(T / dt)
    t = np.linspace(0, T, n_steps + 1)
    
    #random Z (to compare the EM and Milstein with same Z)
    if Z is None: Z = np.random.normal(0, 1, size=(n_steps, num_paths))
    elif Z.shape != (n_steps, num_paths): raise ValueError("Provided Z array has incorrect shape")

    #same as before, but basically multiple times (1000 times for example)
    S = np.zeros((n_steps + 1, num_paths))
    S[0, :] = s0 #initial price for ALL the prices
    
    sqrt_dt = np.sqrt(dt)

    for i in range(n_steps):
        S_prev = S[i, :] 
        Z_step = Z[i, :] #fixed Z defined twice, so breaking the other graphs and errors calc

        S[i+1, :] = S_prev + (sigma * S_prev * Z_step * sqrt_dt) + (mu * S_prev * dt) #same formula as above but we vectorized it

    return t, S


def simulate_gbm_milstein_vectorized(s0, mu, sigma, T, dt, num_paths, Z = None):
    #same variables as before !
    n_steps = int(T / dt)
    t = np.linspace(0, T, n_steps + 1)

    S = np.zeros((n_steps + 1, num_paths))
    S[0, :] = s0

    #we generate a random z if not provided
    if Z is None: Z = np.random.normal(0, 1, size=(n_steps, num_paths))
    elif Z.shape != (n_steps, num_paths): raise ValueError("Provided Z array has incorrect shape")

    sqrt_dt = np.sqrt(dt)

    #loop through time steps
    for i in range(n_steps):
        S_prev = S[i, :] #current step
        Z_step = Z[i, :] #random shock with Z on every i
        
        dW = Z_step * sqrt_dt

        #Milstein formula here (in diff parts, basically = actual price + euler term + milstein correction)
        euler_term = (mu * S_prev * dt) + (sigma * S_prev * dW)
        
        #correction term when using Z : 0.5 * sigma^2 * S_prev * dt * (Z_step**2 - 1)
        milstein_correction = 0.5 * (sigma**2) * S_prev * dt * (Z_step**2 - 1)
        
        #then price update
        S[i+1, :] = S_prev + euler_term + milstein_correction

    return t, S


def simulate_ou_em_vectorized(x0, theta, kappa, sigma, T, dt, num_paths, Z=None):
    #It simulates multiple paths of the Ornstein-Uhlenbeck process using the Euler-Maruyama process in a vectorized manner

    #x0 : initial value
    #theta : speed of mean reversion (theta > 0).
    #kappa : long-term mean level
    #sigma : volatility coefficient


    n_steps = int(T / dt)
    t = np.linspace(0, T, n_steps + 1)

    X = np.zeros((n_steps + 1, num_paths))
    X[0, :] = x0

    if Z is None:
        Z = np.random.normal(0, 1, size=(n_steps, num_paths))
    elif Z.shape != (n_steps, num_paths):
        raise ValueError("Provided Z array has incorrect shape")

    sqrt_dt = np.sqrt(dt)

    #loop through time steps (vectorized across paths)
    for i in range(n_steps):
        X_prev = X[i, :] #current
        Z_step = Z[i, :] #random shock for all prices

        #dW calc for the step
        dW = Z_step * sqrt_dt

        #Euler-Maruyama formula for OU:
        #dX = theta*(kappa - X)*dt + sigma*dW
        X[i+1, :] = X_prev + theta * (kappa - X_prev) * dt + sigma * dW

        #and unlike GBM, OU process can become negative !

    return t, X


def simulate_cir_em_vectorized(x0, theta, kappa, sigma, T, dt, num_paths, Z=None):
    #simulates CIR process using Euler-Maruyama with Full Truncation
    n_steps = int(T / dt)
    t = np.linspace(0, T, n_steps + 1)

    X = np.zeros((n_steps + 1, num_paths))
    X[0, :] = x0

    #generate random numbers if not provided
    if Z is None:
        Z = np.random.normal(0, 1, size=(n_steps, num_paths))
    elif Z.shape != (n_steps, num_paths):
        raise ValueError("Provided Z array has incorrect shape")

    sqrt_dt = np.sqrt(dt)
    
    #precompute constants
    theta_kappa_dt = theta * kappa * dt
    one_minus_theta_dt = 1.0 - theta * dt

    #loop through time steps
    for i in range(n_steps):
        X_prev = X[i, :]    
        Z_step = Z[i, :]    
        dW = Z_step * sqrt_dt

        #ensure non-negative value inside sqrt (Full Truncation)
        sqrt_X_prev_safe = np.sqrt(np.maximum(0, X_prev)) 

        #apply Euler-Maruyama formula for CIR
        #dX = theta*(kappa - X)*dt + sigma*sqrt(X)*dW
        #X[i+1] = X_prev + theta*kappa*dt - theta*X_prev*dt + sigma*sqrt(max(0,X_prev))*dW
        #X[i+1] = theta_kappa_dt + X_prev*(1 - theta*dt) + sigma*sqrt_X_prev_safe*dW
        X[i+1, :] = theta_kappa_dt + X_prev * one_minus_theta_dt + sigma * sqrt_X_prev_safe * dW
        
        #explicitly enforce positivity after step if desired (alternative scheme)
        #X[i+1, :] = np.maximum(0, X[i+1, :]) #Reflection method essentially

    return t, X


def simulate_cir_milstein_vectorized(x0, theta, kappa, sigma, T, dt, num_paths, Z=None):
    #simulates CIR process using Milstein scheme
    #includes positivity enforcement after the step
    n_steps = int(T / dt)
    t = np.linspace(0, T, n_steps + 1)

    X = np.zeros((n_steps + 1, num_paths))
    X[0, :] = x0

    #generate random numbers if not provided
    if Z is None:
        Z = np.random.normal(0, 1, size=(n_steps, num_paths))
    elif Z.shape != (n_steps, num_paths):
        raise ValueError("Provided Z array has incorrect shape")

    sqrt_dt = np.sqrt(dt)
    
    #precompute constants
    theta_kappa_dt = theta * kappa * dt
    one_minus_theta_dt = 1.0 - theta * dt
    milstein_coeff = 0.25 * (sigma**2) * dt #constant part of correction

    #loop through time steps
    for i in range(n_steps):
        X_prev = X[i, :]    
        Z_step = Z[i, :]    
        dW = Z_step * sqrt_dt

        #ensure non-negative value inside sqrt for diffusion term calc
        sqrt_X_prev_safe = np.sqrt(np.maximum(0, X_prev)) 

        #calculate EM part first
        em_step = theta_kappa_dt + X_prev * one_minus_theta_dt + sigma * sqrt_X_prev_safe * dW
        
        #calculate Milstein correction term
        milstein_correction = milstein_coeff * (Z_step**2 - 1)
        
        #add correction and apply positivity enforcement
        X[i+1, :] = np.maximum(0, em_step + milstein_correction) #full truncation/reflection after step

    return t, X


def simulate_merton_jd_vectorized(s0, mu, sigma, lambda_jump, k_jump, nu_jump, T, dt, num_paths, Z=None):
    #simulates Merton Jump-Diffusion model using Euler scheme + Poisson jumps
    n_steps = int(T / dt)
    t = np.linspace(0, T, n_steps + 1)

    S = np.zeros((n_steps + 1, num_paths))
    S[0, :] = s0

    #generate standard normal random numbers for diffusion if not provided
    if Z is None:
        Z = np.random.normal(0, 1, size=(n_steps, num_paths))
    elif Z.shape != (n_steps, num_paths):
        raise ValueError("Provided Z array has incorrect shape")

    #calculate adjusted drift for continuous part
    drift_adj = mu - lambda_jump * k_jump 
    
    #jump parameters for ln(Y) ~ N(m, nu^2)
    m_jump = np.log(1 + k_jump) - 0.5 * nu_jump**2
    
    sqrt_dt = np.sqrt(dt)

    #loop through time steps
    for i in range(n_steps):
        S_prev = S[i, :]    
        Z_step = Z[i, :]    
        dW = Z_step * sqrt_dt

        #1. continuous diffusion part (using adjusted drift)
        S_cont = S_prev + (drift_adj * S_prev * dt) + (sigma * S_prev * dW)
        
        #2. jump part - simulate number of jumps in dt
        poisson_mean = lambda_jump * dt
        dN = np.random.poisson(poisson_mean, size=num_paths)
        
        #initialize jump multiplier to 1 (no jump effect)
        jump_multiplier = np.ones(num_paths)
        
        #find paths that have jumps
        jump_indices = np.where(dN > 0)[0]
        
        if len(jump_indices) > 0:
            #only simulate jumps for paths that need them
            num_jumps_this_step = dN[jump_indices] #array of jump counts (1, 2, 3...) for jumping paths
            total_jumps_to_sim = np.sum(num_jumps_this_step) #total number of Y values needed
            
            #simulate all needed log-jump sizes at once
            ln_Y = np.random.normal(loc=m_jump, scale=nu_jump, size=total_jumps_to_sim)
            
            #apply jumps cumulatively - this part is tricky to vectorize perfectly if dN > 1 often
            #simpler approach: loop through max jumps, or calculate total jump factor
            
            #approach: calculate total jump factor using exp(sum(lnY))
            total_lnY = np.zeros(len(jump_indices))
            current_jump_idx = 0
            for k, num_j in enumerate(num_jumps_this_step):
                #sum the log jumps for path k (which has num_j jumps)
                total_lnY[k] = np.sum(ln_Y[current_jump_idx : current_jump_idx + num_j])
                current_jump_idx += num_j
                
            #calculate the multiplier exp(sum(lnY)) and place it in the correct indices
            jump_multiplier[jump_indices] = np.exp(total_lnY)
            
        #3. apply jump multiplier
        S[i+1, :] = S_cont * jump_multiplier
        
        #optional positivity enforcement (though jumps can make it negative if k is very negative)
        S[i+1, :] = np.maximum(1e-8, S[i+1, :]) #small floor instead of zero

    return t, S