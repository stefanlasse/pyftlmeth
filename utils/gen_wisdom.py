import FourierMethods as fm
import numpy
import sys

"""
Generating wisdom for FourierMethods.py:

minArraySize: minimum number of elements in array
minArraySize: maximum number of elements in array
linearStep:   step size for last generation step.

CAUTION! Depending on how big maxArraySize is 
         and how dense the linear spaced array sizes
         are this program can run up to several days.
"""

minArraySize = int(1.0E3)
maxArraySize = int(1.0E7)
linearStep = int(1.0E3)


#--------------------------------------------------------------------------
def primesfrom2to(n=6):
    """ Input n>=6, Returns a array of primes, 2 <= p < n """
    sieve = numpy.ones(n/3 + (n%6==2), dtype=numpy.bool)
    for i in xrange(1,int(n**0.5)/3+1):
        if sieve[i]:
            k=3*i+1|1
            sieve[k*k/3::2*k] = False
            sieve[k*(k-2*(i&1)+4)/3::2*k] = False
    return numpy.r_[2,3,((3*numpy.nonzero(sieve)[0][1:]+1)|1)]

#--------------------------------------------------------------------------
def floorLog2(n=2):
	"""Finds the largest power of 2 which is smaller than the array size."""
	last = n
	n &= n - 1
	while n:
		last = n
		n &= n - 1
	return int(numpy.log2(last))

#--------------------------------------------------------------------------
def generateWisdomFor(numbers):
	for i in numbers:
		sys.stdout.write("\r --> gen wisdom for %d points"%(i))
		sys.stdout.flush()
		fft = fm.FourierTransform(i)
		del fft
		ifft = fm.InverseFourierTransform(i)
		del ifft

#==========================================================================
# generate wisdom for prime number long arrays
print "Prime number long arrays"
primes = primesfrom2to(5000)
generateWisdomFor(primes)

# generate wisdom for power of two number long arrays
print "\nPower of two long arrays"
powerTwo = [int(2**i) for i in range(2, floorLog2(maxArraySize)+1)]
generateWisdomFor(powerTwo)

# generate wisdom for linear spaced array sizes
print "\nLinear spaced arrays"
linearSpaced = range(minArraySize, maxArraySize+linearStep, linearStep)
generateWisdomFor(linearSpaced)

print "\n\nDone.\n"