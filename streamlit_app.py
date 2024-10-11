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

# Step 5: Calculate total open interest
combined_oi['total_open_interest'] = combined_oi['calls_openInterest'] + combined_oi['puts_openInterest']

# Step 6: Find the 5 strikes with the highest open interest
top_strikes = combined_oi.nlargest(5, 'total_open_interest')

# Step 7: Fetch NVDA stock price data for the past 1 month and prepare for plotting
nvda_price = nvda.history(period="1mo")

# Plotting
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(nvda_price.index, nvda_price['Close'], label="NVDA Close Price", color="blue")
ax.set_title(f'NVDA Stock Price with Top 5 Open Interest Strikes (Expiry: {expiry_date})')
ax.set_xlabel("Date")
ax.set_ylabel("Price")
ax.yaxis.tick_right()  # Move y-axis ticks to the right
ax.yaxis.set_label_position("right")  # Move y-axis label to the right

# Mark the top 5 strike prices on the chart and label them
for i, row in top_strikes.iterrows():
    strike_price = row['strike']
    ax.axhline(y=strike_price, color='red', linestyle='--', label=f"Strike {strike_price} (OI: {int(row['calls_openInterest'] + row['puts_openInterest'])})")
    ax.text(nvda_price.index[-1], strike_price, f"{strike_price}", color='red', verticalalignment='bottom', fontsize=10)

# Add legend to the plot
ax.legend()

# Display plot in Streamlit
st.pyplot(fig)

# Output the top 5 strikes and their total open interest, gamma, calls, and puts
st.subheader("Top 5 Strikes with Highest Open Interest:")

# Create a list to hold the formatted output
top_strikes_output = []

for i, row in top_strikes.iterrows():
    total_open_interest = row['total_open_interest']
    calls_oi = row['calls_openInterest']
    puts_oi = row['puts_openInterest']
    
    # Calculate percentages
    calls_percentage = (calls_oi / total_open_interest * 100) if total_open_interest > 0 else 0
    puts_percentage = (puts_oi / total_open_interest * 100) if total_open_interest > 0 else 0
    total_gamma = row['calls_gamma'] + row['puts_gamma']
    
    # Append to output list
    top_strikes_output.append({
        'Strike': row['strike'],
        'Calls (%)': f"{calls_percentage:.2f}%",
        'Puts (%)': f"{puts_percentage:.2f}%",
        'Total Gamma': f"{total_gamma:.6f}"
    })

# Display the output as an array
st.write(top_strikes_output)
