import yfinance as yf
import streamlit as st
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker  # Import ticker for scientific notation

# for  hypothesis testing
import scipy.stats as stats
import numpy as np

st.write("""
         # Stock Price Time Series App

         Using the _yfinance_ package we import the price and volume data of GOOGLE!
         This is an exploratory study on the characteristics of the Google stock between '2010-01-01' and '2020-12-31'.

         """)

tickerSymbol = 'GOOGL'
tickerData = yf.Ticker(tickerSymbol)
tickerDF = tickerData.history(period='1d',
                              start='2010-01-01',
                              end='2020-12-31')

st.write("""
         ## GOOGLE Closing Prices

         - During this period Google stock has risen steadily.
         - However, there was a significant dip in the year 2020. This was probably a result of COVID-19 related market stresses.
         """)
st.line_chart(tickerDF.Close)

st.write("""
         ## GOOGLE Trading Volumes

         - Before 2015 the trading volumes were generally higher and extreme peaks were more common.
         - However, the second part of the period under review has different characteristics.
         - This sugests that there was a regime change in the market.
         """)
st.line_chart(tickerDF.Volume)

st.write("""
         ### Hypothesis testing

         - Out of interest we will test the hypothesis that the volumes after 2014 are lower than volums before 2015.
         - The histograms seem to support the idea that trading volumes are lower after 2014
         """)


split_year = 2015
tickerDF.reset_index(inplace=True)

# Create two DataFrames
df_before = tickerDF[tickerDF["Date"].dt.year < split_year]
df_after = tickerDF[tickerDF["Date"].dt.year >= split_year]

# Sample DataFrames (assuming 'value' column exists in both)
fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)  # 1 row, 2 columns

# Histogram for df_before
axes[0].hist(df_before["Volume"], bins=20, color='blue', alpha=0.7)
axes[0].set_title("Volumes Before 2015")
axes[0].set_xlabel("Volume")
axes[0].set_ylabel("Frequency")
axes[0].grid()

# Histogram for df_after
axes[1].hist(df_after["Volume"], bins=20, color='red', alpha=0.7)
axes[1].set_title("Volumes After and Including 2015")
axes[1].set_xlabel("Volume")
axes[1].grid()

# Apply scientific notation formatter
for ax in axes:
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter(useMathText=True))
    ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))  # Force scientific notation

# Adjust layout
plt.tight_layout()

#display in streamlit
st.pyplot(fig)


stat, p1 = stats.levene(df_before["Volume"], df_after["Volume"])
st.write("""
         #### Test for Equal Variances

         - Lets determine if the two group have significantly different variances
         - Hypothesis:
            - $H_0:$ The variances before and after 2015 are the same
            - $H_1$: The variances are not the same
         - Significance level:
            - $alpha$ = 0.01
         - Conclusion
            - There is very strong evidence to suggest that the variances of the two time periods are different.\n
                 Test statistic =395.135, and p-value = 2.692*$10^{-8}$
         """)


tstat, p2 = stats.ttest_ind(df_before["Volume"], df_after["Volume"], equal_var = False, alternative='greater')

st.write("""
         #### Test for Equal  (Median) Volumes

         - Lets determine if the two group have significantly different variances
         - Hypothesis:
            - $H_0:$ Median volumes before and after 2015 are the same
            - $H_1$: The medians are not the same
         - Significance level:
            - $alpha$ = 0.01
        - Conclusion:
            - There is very strong evidence to suggest that the median volumes of the two time periods are different.
         """)
fString = f"""- Test statistic ={tstat: .3f}, and p-value = {p2: .3f}."""
st.write(fString)

tickerDF['Returns'] = tickerDF.Close.pct_change().dropna()
st.write("""
         ## GOOGLE Return Characteristics

         - Market returns exhibit typical characteristics.
         - We will analyse them below

         ### Returns Histogram
         """)

from matplotlib import colors
#from matplotlib.ticker import PercentFormatter

fig, ax = plt.subplots()
N, bins, patches = ax.hist(tickerDF.Returns, bins=50, density=True)
fracs = N / N.max()

# we need to normalize the data to 0..1 for the full range of the colormap
norm = colors.Normalize(fracs.min(), fracs.max())

# Now, we'll loop through our objects and set the color of each accordingly
for thisfrac, thispatch in zip(fracs, patches):
    color = plt.cm.viridis(norm(thisfrac))
    thispatch.set_facecolor(color)

#ax.yaxis.set_major_formatter(PercentFormatter(xmax=1))
ax.grid()

st.pyplot(fig)
st.write("""
         ### Volatility Clustering
         """)
fig, ax = plt.subplots()
ax.plot(tickerDF.Date,tickerDF.Returns)
st.pyplot(fig)

def rolling_returns(data1, N):
    return data1.rolling(N).mean()
def rolling_vol(data1, N):
    return data1.rolling(N).std() * np.sqrt(N)
def rolling_corr(data1, data2, N):
    return data1.rolling(N).corr(data2)

tickerDF['Returns_Mean'] = rolling_returns(tickerDF.Returns, 252)
tickerDF['Returns_Vol'] = rolling_vol(tickerDF.Returns, 252)

tickerDF.dropna()
tickerDF['Returns_Vol_Corr'] = rolling_corr(tickerDF.Returns_Mean, tickerDF.Returns_Vol, 252)
st.write("""
         ### Leverage Effect and Stochastic Volatility
         """)
fig, (ax,ax1) = plt.subplots(2,1)
# Plotting the first dataset
ax.plot(tickerDF.Date, tickerDF.Returns_Mean, 'b-', label='Rolling Mean Ret')
ax.set_xlabel('Date')
ax.set_ylabel('')
ax.tick_params('y', colors='b')

# Creating a second y-axis
ax2 = ax.twinx()
ax2.plot(tickerDF.Date, tickerDF.Returns_Vol, 'r:', label='Rolling Volatility')

ax2.set_ylabel('Y2-axis', color='r')
ax2.tick_params('y', colors='r')

# Adding legends
fig.tight_layout()
ax.legend(loc='upper right')
ax2.legend(loc='lower right')
ax.legend()

ax1.plot(tickerDF.Date, tickerDF.Returns_Vol_Corr, 
         'g--', label='Rolling Correlation')
# Displaying the plot in Streamlit
st.pyplot(fig)