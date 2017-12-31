from alpha_vantage.techindicators import TechIndicators
from alpha_vantage.timeseries import TimeSeries
import matplotlib.pyplot as plt

#ti = TechIndicators(key='XNL4', output_format='pandas')
#data, meta_data = ti.get_bbands(symbol='MSFT', interval='daily', time_period=5, series_type='close')
#data.plot()
#plt.title('BBbands indicator for  MSFT stock (1 day)')
#plt.show()

ts = TimeSeries(key='XNL4', output_format='pandas')
data, meta_data = ts.get_daily_adjusted(symbol='MSFT', outputsize='full')
data['close'].plot()
plt.title('Intraday Times Series for the MSFT stock (1 day)')
plt.show()