import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

# Black-Scholes Gamma Calculation Function
def calculate_gamma(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    return gamma

# Streamlit app
st.title("NVDA 0DTE Options Analysis")

# Step 1: Fetch NVDA option chain for today
ticker = 'NVDA'
nvda = yf.Ticker(ticker)

# Get current stock price
current_price = nvda.history(period="1d")['Close'].iloc[0]

# Get today's date options (0DTE) by selecting the nearest expiry date
options = nvda.options
expiry_date = options[0]
option_chain = nvda.option_chain(expiry_date)

# Step 2: Get calls and puts
calls = option_chain.calls
puts = option_chain.puts

# Step 3: Calculate gamma for each option using the Black-Scholes formula
risk_free_rate = 0.05  # 5% risk-free rate
time_to_expiration = 1 / 365  # 0DTE means the expiration is in one day

# Calculate gamma for calls and puts
calls['gamma_manual'] = calls.apply(
    lambda x: calculate_gamma(
        S=current_price, 
        K=x['strike'], 
        T=time_to_expiration, 
        r=risk_free_rate, 
        sigma=x['impliedVolatility']
    ), axis=1
)

puts['gamma_manual'] = puts.apply(
    lambda x: calculate_gamma(
        S=current_price, 
        K=x['strike'], 
        T=time_to_expiration, 
        r=risk_free_rate, 
        sigma=x['impliedVolatility']
    ), axis=1
)

# Step 4: Aggregate open interest by strike
calls_oi = calls.groupby('strike').agg({'openInterest': 'sum', 'gamma_manual': 'sum'}).reset_index()
calls_oi.rename(columns={'openInterest': 'calls_openInterest', 'gamma_manual': 'calls_gamma'}, inplace=True)

puts_oi = puts.groupby('strike').agg({'openInterest': 'sum', 'gamma_manual': 'sum'}).reset_index()
puts_oi.rename(columns={'openInterest': 'puts_openInterest', 'gamma_manual': 'puts_gamma'}, inplace=True)

# Merge calls and puts open interest and gamma
combined_oi = pd.merge(calls_oi, puts_oi, on='strike', how='outer').fillna(0)

# Step 5: Calculate total open interest and get top 3 strikes with highest OI
combined_oi['total_open_interest'] = combined_oi['calls_openInterest'] + combined_oi['puts_openInterest']
top_3_strikes = combined_oi.nlargest(3, 'total_open_interest')[['strike', 'calls_openInterest', 'puts_openInterest', 'calls_gamma', 'puts_gamma']]

# Step 6: Fetch NVDA stock price data for the past 1 month and prepare for plotting
nvda_price = nvda.history(period="1mo")

# Plotting
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(nvda_price.index, nvda_price['Close'], label="NVDA Close Price", color="blue")
ax.set_title(f'NVDA Stock Price with Top 3 Open Interest Strikes (Expiry: {expiry_date})')
ax.set_xlabel("Date")
ax.set_ylabel("Price")
ax.yaxis.tick_right()  # Move y-axis ticks to the right
ax.yaxis.set_label_position("right")  # Move y-axis label to the right

# Mark the top 3 strike prices on the chart
for i, row in top_3_strikes.iterrows():
    ax.axhline(y=row['strike'], color='red', linestyle='--', label=f"Strike {row['strike']} (OI: {int(row['calls_openInterest'] + row['puts_openInterest'])})")

# Add legend to the plot
ax.legend()

# Display plot in Streamlit
st.pyplot(fig)

# Output the top 3 strikes and their total open interest, gamma, calls, and puts
st.subheader("Top 3 Strikes with the Most Open Interest:")
for i, row in top_3_strikes.iterrows():
    total_gamma = row['calls_gamma'] + row['puts_gamma']
    st.write(f"**Strike:** {row['strike']}, **Calls OI:** {int(row['calls_openInterest'])}, **Puts OI:** {int(row['puts_openInterest'])}, "
             f"**Calls Gamma:** {row['calls_gamma']:.6f}, **Puts Gamma:** {row['puts_gamma']:.6f}, **Total Gamma:** {total_gamma:.6f}")
