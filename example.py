# %%

import matplotlib.pyplot as plt
import pandas as pd
from pmdarima.arima.utils import ndiffs
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

plt.rcParams.update({'figure.figsize': (9, 7), 'figure.dpi': 120})

# Import data
df = pd.read_csv('hpa.csv', names=['time', 'value'], header=None)

minTime = min(df.time)
maxTime = max(df.time)

print('min time', minTime)
print('max time', maxTime)

## Adf Test
print('adf', ndiffs(df.value, test='adf'))

# KPSS test
print('kpss', ndiffs(df.value, test='kpss'))

# PP test:
print('pp', ndiffs(df.value, test='pp'))

# Original Series
fig, axes = plt.subplots(3, 2)
axes[0, 0].set_xlim(left=minTime, right=maxTime)
axes[0, 0].plot(df.time, df.value)
axes[0, 0].set_title('Original Series')
plot_acf(df.value, ax=axes[0, 1])

# 1st Differencing
axes[1, 0].set_xlim(left=minTime, right=maxTime)
axes[1, 0].plot(df.time, df.value.diff())
axes[1, 0].set_title('1st Order Differencing')
plot_acf(df.value.diff().dropna(), ax=axes[1, 1])

# 2nd Differencing
axes[2, 0].set_xlim(left=minTime, right=maxTime)
axes[2, 0].plot(df.time, df.value.diff().diff())
axes[2, 0].set_title('2nd Order Differencing')
plot_acf(df.value.diff().diff().dropna(), ax=axes[2, 1])

plt.show()

### Partial autocorrelation

# PACF plot of 1st differenced series
plt.rcParams.update({'figure.figsize': (9, 3), 'figure.dpi': 120})

fig2, axes2 = plt.subplots(1, 2)
axes2[0].set_xlim(left=minTime, right=maxTime)
axes2[0].plot(df.time, df.value)
axes2[0].set_title('No Differencing')
plot_pacf(df.value, ax=axes2[1])

plt.show()
