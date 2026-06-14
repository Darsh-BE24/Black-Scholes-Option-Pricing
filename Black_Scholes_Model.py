import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq
import matplotlib.pyplot as plt

# --- Black-Scholes Model Functions ---
def black_scholes_price(S, K, T, r, sigma, option_type='call'):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == 'call':
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == 'put':
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    else:
        raise ValueError("option_type must be 'call' or 'put'")
    return price

# --- Implied Volatility Estimation ---
def implied_volatility(S, K, T, r, market_price, option_type='call'):
    def objective(sigma):
        return black_scholes_price(S, K, T, r, sigma, option_type) - market_price
    
    try:
        return brentq(objective, 1e-6, 5)
    except ValueError:
        return np.nan

# --- Example Usage ---
if __name__ == "__main__":
    # Example parameters
    S = 100        # Spot price
    K = 100        # Strike price
    T = 1          # Time to maturity (in years)
    r = 0.05       # Risk-free interest rate
    sigma = 0.2    # Volatility

    # Compute theoretical option prices
    call_price = black_scholes_price(S, K, T, r, sigma, 'call')
    put_price = black_scholes_price(S, K, T, r, sigma, 'put')

    print(f"Call Price: {call_price:.4f}")
    print(f"Put Price:  {put_price:.4f}")

    # Estimate implied volatility from market price
    market_call_price = 10
    iv = implied_volatility(S, K, T, r, market_call_price, 'call')
    print(f"Implied Volatility (call): {iv:.4f}")

    # Sensitivity plot - Call price vs Volatility
    vol_range = np.linspace(0.01, 1, 100)
    prices = [black_scholes_price(S, K, T, r, vol, 'call') for vol in vol_range]
    
    plt.figure(figsize=(8, 5))
    plt.plot(vol_range, prices, label='Call Price')
    plt.title('Call Option Price vs Volatility')
    plt.xlabel('Volatility')
    plt.ylabel('Call Option Price')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()