import sys
from pyftlmeth import FourierMethods as c
import numpy as np
import matplotlib.pyplot as plt

numPoints = int(10000)
dataA=np.zeros(numPoints)

# test auto correlation
cData = dataA
mean = 20
cDataN = cData + np.random.poisson(mean, cData.size)
cDataN = np.roll(cDataN, np.random.randint(0, cDataN.size))
cDataN += np.roll(cDataN, np.random.randint(0, cDataN.size))

autoCorrTS = c.TimeSeries(cDataN, fs=1.0)

autocorr = c.AutoCorrelation(size=autoCorrTS.data.size, dtype=autoCorrTS.data.dtype.name)

result = autocorr(autoCorrTS)
npResult = np.correlate(autoCorrTS.data, autoCorrTS.data, 'full')

plt.plot(autoCorrTS.time, autoCorrTS.data)
plt.plot(result.time[result.time.size//2:], result.data[result.data.size//2:])
plt.plot(result.time[result.time.size//2:], npResult[npResult.size//2:])
plt.title("auto")
plt.show()
