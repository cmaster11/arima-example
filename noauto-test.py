# %%

import matplotlib.pyplot as plt
import pandas as pd
import math
import pmdarima as pm
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima_model import ARIMA

# Import data
df = pd.read_csv('hpa.csv', names=['time', 'value'], header=None)

minTime = min(df.time)
maxTime = max(df.time)

train = df.value[:math.floor(3/4*len(df.value))+1]
test = df.value[math.floor(3/4*len(df.value)):]

model = ARIMA(train, order=(0, 2, 1))
fitted = model.fit(disp=-1)

# Forecast
fc, se, conf = fitted.forecast(len(df.value) - math.floor(3/4*len(df.value)), alpha=0.05)  # 95% conf

# Make as pandas series
fc_series = pd.Series(fc, index=test.index)
lower_series = pd.Series(conf[:, 0], index=test.index)
upper_series = pd.Series(conf[:, 1], index=test.index)

# Plot
plt.figure(figsize=(12,5), dpi=100)
plt.plot(train, label='training')
plt.plot(test, label='actual')
plt.plot(fc_series, label='forecast')
plt.fill_between(lower_series.index, lower_series, upper_series,
                 color='k', alpha=.15)
plt.title('Forecast vs Actuals')
plt.legend(loc='upper left', fontsize=8)
plt.show()

plt.rcParams.update({'figure.figsize': (9, 7), 'figure.dpi': 120})