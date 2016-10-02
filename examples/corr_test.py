
import FourierMethods as c
import numpy as np
import matplotlib.pyplot as plt
from time import time

numPoints = int(16651)
iterator = 5


dataA=np.zeros(numPoints)
dataB=np.zeros(numPoints)


start=400
stop =600
dataType='float64'
dataA[start:stop] = np.ones(stop-start, dtype=dataType)
dataB[start:stop] = np.linspace(1.0, 0.0, stop-start, dtype=dataType)



# test cross correlation
corr = c.CrossCorrelation(arraySize=dataA.size, realOut=True, dtype=dataType)

for i in range(iterator):
	dataB = np.roll(dataB, np.random.randint(0, dataB.size))
	result = corr(dataA, dataB)
	npResult = np.correlate(dataA, dataB, 'full')

	plt.plot(dataA)
	plt.plot(dataB)
	plt.plot(result)
	plt.plot(npResult)
	plt.title("cross")
	plt.show()



# test auto correlation
cData = dataA
mean = 20
cDataN = cData + np.random.poisson(mean, cData.size)
autocorr = c.AutoCorrelation(arraySize=cDataN.size, realOut=True, dtype=dataType)


for i in range(iterator):
	cDataN = np.roll(cDataN, np.random.randint(0, cDataN.size))
	cDataN += np.roll(cDataN, np.random.randint(0, cDataN.size))
	result = autocorr(cDataN)
	npResult = np.correlate(cDataN, cDataN, 'full')

	plt.plot(cData)
	plt.plot(cDataN)
	plt.plot(result[result.size/2:])
	plt.plot(npResult[npResult.size/2:])
	plt.title("auto")
	plt.show()

del autocorr

exit

cDataN = np.asarray(np.random.poisson(mean, cData.size), dtype='float64')
autocorr = c.AutoCorrelation(arraySize=cDataN.size, realOut=True, dtype=dataType)
result = autocorr(cDataN)
npResult = np.correlate(cDataN, cDataN, 'same')

#plt.plot(cData)
plt.plot(cDataN)
plt.plot(result)
plt.plot(npResult+1)
plt.show()





