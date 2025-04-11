# SDE Simulation, Analysis, and Application in Finance

## Overview

This project explores the simulation of stochastic processes commonly used to model the dynamics of financial assets. It involves implementing various Stochastic Differential Equation (SDE) models, comparing numerical simulation techniques (Euler-Maruyama and Milstein), analyzing their accuracy through convergence tests, and applying these simulations to practical areas like option pricing and parameter estimation. A basic strategy backtesting component was also developed to further explore applications.

## Topics & Techniques Covered

*   **SDE Models Simulated:**
    *   Geometric Brownian Motion (GBM)
    *   Ornstein-Uhlenbeck (OU) Process
    *   Cox-Ingersoll-Ross (CIR) Process
    *   Merton's Jump-Diffusion (JD) Model
*   **Numerical Simulation Schemes:**
    *   Euler-Maruyama (EM)
    *   Milstein Scheme
    *   Vectorized implementation using NumPy for efficiency.
*   **Model Validation & Analysis:**
    *   Comparison with Analytical Solutions (GBM, OU distributions)
    *   Strong Convergence Analysis (Order 0.5 vs 1.0)
    *   Weak Convergence Analysis (Order 1.0, noting numerical limits)
    *   Handling model constraints (e.g., positivity in CIR)
*   **Applications:**
    *   Monte Carlo Pricing (European Options on GBM)
    *   Parameter Estimation (Maximum Likelihood Estimation - MLE for OU)
    *   Basic Strategy Backtesting (Dual Moving Average Crossover on SPY)
    *   Parameter Optimization (Heatmap analysis for backtesting)
*   **Tools:** Python, NumPy, SciPy, Pandas, Matplotlib, Seaborn, yfinance

---

## Methodology & Key Findings

### 1. SDE Simulation Engine

Vectorized simulators were implemented in Python for key SDEs:

*   **GBM (`simulate_gbm_..._vectorized`):** Standard model for non-mean-reverting assets like stocks. `dS = μSdt + σSdW`.
*   **OU (`simulate_ou_em_vectorized`):** Mean-reverting process, often used for interest rates or volatility. `dX = θ(κ - X)dt + σdW`. Simulators captured mean reversion towards `κ`.
    *(See Notebook 05 for OU Path Plot & Distribution Plot results confirming visual mean reversion and good match with analytical Normal distribution)*
*   **CIR (`simulate_cir_..._vectorized`):** Mean-reverting process with `sqrt(X)` diffusion, ensuring positivity (under Feller cond.). Used for interest rates. `dX = θ(κ - X)dt + σ√X dW`. Implemented with full truncation EM scheme to handle positivity near zero.
    *(See Notebook 07 for CIR Path Plot results confirming positivity and state-dependent volatility)*
*   **Merton JD (`simulate_merton_jd_vectorized`):** GBM extended with Poisson-driven jumps to model sudden price shocks. `dS/S = (μ - λk)dt + σdW + dJ`.
    *(See Notebook 08 for JD Path & Distribution Plots showing jump discontinuities and fatter tails compared to GBM)*

Two numerical schemes were implemented:

*   **Euler-Maruyama (EM):** Basic, first-order scheme.
*   **Milstein:** Higher-order scheme adding a correction term (`0.5*g*g'*(dW^2-dt)`), expected to improve strong convergence for SDEs with non-constant diffusion coefficients `g(X)`.

### 2. Validation & Convergence Analysis

Simulators were validated quantitatively:

*   **Analytical Distribution Comparison:** For GBM and OU, histograms of simulated final values `S(T)` / `X(T)` matched the known analytical Log-Normal / Normal distributions well.
*   **Strong Convergence:** Measured pathwise error (`E[|X_sim(T) - X_ref(T)|]`) vs `dt` on log-log plots.
    *   **GBM:** Successfully showed EM converging at order ~0.5 and Milstein at order ~1.0, confirming theory. *(See Notebook 03 for GBM Strong Convergence Plot)*
    *   **OU:** Showed EM converging at order ~0.5. Confirmed Milstein reduces to EM (zero correction term) due to constant diffusion `σ`. *(See Notebook 05 for OU Strong Convergence Plot)*
    *   **CIR:** Clearly demonstrated EM converging at order ~0.5, while Milstein achieved order ~1.0, showcasing Milstein's advantage for non-constant diffusion (`σ√X`). *(See Notebook 07 for CIR Strong Convergence Plot)*
*   **Weak Convergence:** Tested for GBM. Showed expected rate (~1.0) for both EM/Milstein, but encountered numerical noise floor limitations at very small `dt`. *(See Notebook 03 for GBM Weak Convergence Plot)*

### 3. Applications

Applied the simulation framework to practical problems:

*   **Monte Carlo Option Pricing:**
    *   Implemented MC pricer for European call options using the GBM simulator under the risk-neutral measure (`drift=r`).
    *   Demonstrated convergence of the MC price estimate to the analytical Black-Scholes price as `num_paths` increased. Showed error decreased approximately as `1/sqrt(num_paths)`. *(See Notebook 04 for MC Price & Error Convergence Plots)*
*   **Maximum Likelihood Estimation (MLE):**
    *   Implemented MLE pipeline (likelihood function + optimizer) to estimate OU parameters (`theta`, `kappa`, `sigma`) from a simulated data path.
    *   Noted practical challenges (discretization bias) affecting estimation accuracy even with high-quality data and different likelihood approaches (exact vs. approximate). *(See Notebook 06 for MLE results summary)*

### 4. Backtesting Extension

Developed and tested a simple trading strategy:

*   **Strategy:** Dual Moving Average (DMA) Crossover on SPY ETF daily data (2010-2025).
*   **Framework:** Implemented vectorized backtesting including signal generation, position logic, log returns, and transaction cost modeling.
*   **Parameter Optimization:** Performed a grid search over short/long window SMA parameters and visualized Sharpe ratio performance using a heatmap. Identified top-performing parameter sets (in-sample).
*   **Performance:** Compared the optimized strategy's equity curve (log scale) against the Buy & Hold benchmark. *(See Notebook 09 for Equity Curve Plot & Heatmap Plot)*

---

## Project Structure

    sde_simulation_project/
    │
    ├── notebooks/         # Analysis and visualization notebooks
    │   ├── 01_gbm_em_simulation.ipynb
    │   ├── ...            # Other notebooks (02 to 09)
    │   └── 09_backtesting_dma.ipynb 
    │
    ├── src/               # Source code
    │   ├── estimation.py  # MLE functions
    │   ├── pricing.py     # Option pricing functions
    │   └── sde_simulator/ # SDE simulation package
    │       └── simulators.py # Core SDE simulators
    │
    ├── .gitignore          
    ├── README.md          # This file
    └── requirements.txt   # Dependencies


---

## Setup & Usage

1.  **Clone the repository:**
    `git clone https://github.com/Far3000-YT/SDE-Simulation-Analysis.git`
    `cd SDE-Simulation-Analysis`

2.  **Create virtual environment (recommended):**
    `python -m venv venv`
    `source venv/bin/activate` (Linux/macOS) or `.\venv\Scripts\activate` (Windows)

3.  **Install dependencies:**
    `pip install -r requirements.txt`

4.  **Run Jupyter Notebooks:**
    `jupyter lab`
    Navigate the `notebooks/` directory to explore analyses.

---

## Technologies Used

Python 3.x, NumPy, SciPy, Pandas, Matplotlib, Seaborn, yfinance

---