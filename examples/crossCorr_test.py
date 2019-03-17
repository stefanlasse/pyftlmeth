from  pyftlmeth import FourierMethods as c
import numpy as np
import matplotlib.pyplot as plt
from time import time

numPoints = int(10000)


dataA=np.zeros(numPoints)
dataB=np.zeros(numPoints)


start=400
stop =600
dataType='float64'
dataA[start:stop] = np.ones(stop-start, dtype=dataType)
dataB[start:stop] = np.linspace(1.0, 0.0, stop-start, dtype=dataType)


dataA = c.TimeSeries(dataA, fs=1.0)
dataB = c.TimeSeries(np.roll(dataB, np.random.randint(0, dataB.size)), fs=1.0)


# test cross correlation
corr = c.CrossCorrelation(size=dataA.data.size, dtype=dataA.data.dtype.name)

result = corr(dataA, dataB)
npResult = np.correlate(dataA.data, dataB.data, 'full')

plt.plot(dataA.time, dataA.data)
plt.plot(dataB.time, dataB.data)
plt.plot(result.time, result.data)
plt.plot(result.time, npResult)
plt.title("cross")
plt.show()
