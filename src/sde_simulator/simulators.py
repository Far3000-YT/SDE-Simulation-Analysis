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


def simulate_gbm_em_vectorized(s0, mu, sigma, T, dt, num_paths):
    n_steps = int(T / dt)
    t = np.linspace(0, T, n_steps + 1)
    
    #same as before, but basically multiple times (1000 times for example)
    S = np.zeros((n_steps + 1, num_paths))
    S[0, :] = s0 #initial price for ALL the prices

    Z_all = np.random.normal(0, 1, size=(n_steps, num_paths)) 
    
    sqrt_dt = np.sqrt(dt)

    for i in range(n_steps):
        S_prev = S[i, :] 
        Z_step = Z_all[i, :] 

        S[i+1, :] = S_prev + (sigma * S_prev * Z_step * sqrt_dt) + (mu * S_prev * dt) #same formula as above but we vectorized it

    return t, S


def simulate_gbm_milstein_vectorized(s0, mu, sigma, T, dt, num_paths, Z=None):
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