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