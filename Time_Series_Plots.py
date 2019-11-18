import pandas as pd
from matplotlib import pyplot
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose

July = pd.read_excel('ActiveSoybeanContractsforJuly2020.CSV.xlsx', header=3)
May = pd.read_excel('ActiveSoybeanContractsForMay2020.CSV.xlsx', header=3)
March = pd.read_excel('ActiveSoybeanContractsForMarch2020.CSV.xlsx', header=3)

for i in [July, May, March]:
    i['Date'] = [ele.date() for ele in i['Date']]

plot_acf(July['Close'])
pyplot.show()

plot_pacf(March['Close'], lags=50)
pyplot.show()

#result = seasonal_decompose(July['Close'], model='additive', freq=5)
#result.plot()
#pyplot.show()

result = seasonal_decompose(July['Close'], model='additive', freq=150)
result.plot()
pyplot.show()
