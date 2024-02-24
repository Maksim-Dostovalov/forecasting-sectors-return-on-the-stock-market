import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import STL
import matplotlib.pyplot as plt

data = pd.read_csv('./data/data_indexes_moex.csv', index_col=0)
data.index = pd.to_datetime(data.index)

ts = data['MOEXMM'].interpolate(method='time')

# Преобразование ежедневных данных в месячные, используя значение за месяц
ts_monthly = ts.resample('M').sum()

decompose = STL(ts_monthly)
result = decompose.fit()

trend = result.trend
seasonal = result.seasonal
residual = result.resid

result.plot()
plt.show()
