# %%

import matplotlib.pyplot as plt
import pandas as pd
from pmdarima.arima.utils import ndiffs
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import acf

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

# axes[0, 1].plot(acf(df.value))
axes[0, 1].set_title('ACF')

plot_acf(df.value, ax=axes[0, 1])

plt.show()