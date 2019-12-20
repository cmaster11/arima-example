# %%

import matplotlib.pyplot as plt
import pandas as pd
import pmdarima as pm
from statsmodels.tsa.stattools import adfuller

# Import data
df = pd.read_csv('hpa.csv', names=['time', 'value'], header=None)

minTime = min(df.time)
maxTime = max(df.time)

result = adfuller(df.value.dropna())
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])

plt.rcParams.update({'figure.figsize': (9, 7), 'figure.dpi': 120})

# Build Model
smodel = pm.auto_arima(
    df.value,
    start_p=1,
    start_q=1,
    test='adf',
    max_p=3, max_q=3,
    m=24 * 6,
    d=None,
    seasonal=True,
    start_P=0,
    D=1,
    trace=False,
    error_action='ignore',
    suppress_warnings=True,
    stepwise=True)

smodel.summary()

# Forecast
n_periods = 5 * 24 * 6
fitted, confint = smodel.predict(n_periods=n_periods, return_conf_int=True)
index_of_fc = pd.date_range(df.value.index[-1], periods=n_periods, freq='10min')

# make series for plotting purpose
fitted_series = pd.Series(fitted, index=index_of_fc)
lower_series = pd.Series(confint[:, 0], index=index_of_fc)
upper_series = pd.Series(confint[:, 1], index=index_of_fc)

# Plot
plt.plot(df.value)
plt.plot(fitted_series, color='darkgreen')
plt.fill_between(lower_series.index,
                 lower_series,
                 upper_series,
                 color='k', alpha=.15)

plt.title("SARIMA - Final Forecast")
plt.show()
