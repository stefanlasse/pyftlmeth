
#==============================================================================
# FourierMethods.py
#
# Copyright (C) 2016 Stefan Lasse, lasse.stefan@gmx.de
#
# This software may be modified and distributed under the terms
# of the MIT license. See the LICENSE file for details.
#==============================================================================

import numpy as np
import pyfftw
import pickle
import sys
from os.path import expanduser, isdir
from os import mkdir, access, W_OK

from TimeSeries import TimeSeries
from Spectrum import Spectrum

#==============================================================================
class FourierMethodBase(object):
	#--------------------------------------------------------------------------
	def __init__(self):
		pass

	#--------------------------------------------------------------------------
	def __del__(self):
		pass

	#--------------------------------------------------------------------------
	def floorLog2(self, n):
		"""Finds the largest power of 2 which is smaller than the array size."""
		last = n
		n &= n - 1
		while n:
			last = n
			n &= n - 1
		return int(np.log2(last))

	#--------------------------------------------------------------------------
	def getWisdom(self):
		"""Returns the current wisdom object of the fftw library."""
		return pyfftw.export_wisdom()

	#--------------------------------------------------------------------------
	def setWisdom(self, wisdom):
		"""Loads the wisdom object parameter to the fftw library."""
		pyfftw.import_wisdom(wisdom)

	#--------------------------------------------------------------------------
	def forgetWisdom(self):
		"""Resets the internal FFTW scheme plan."""
		pyfftw.forget_wisdom()

	#--------------------------------------------------------------------------
	def saveWisdom(self):
		"""Saves the FFTW scheme plan to a pickle file."""
		wisdom = self.getWisdom()

		# check if pyfftw-directory is present in user's home directory
		home = expanduser("~")
		if 'win' in sys.platform:
			pyfftwDir = "\\.pyfftw\\"
		else:
			pyfftwDir = "/.pyfftw/"
		
		directory = home + pyfftwDir

		if not isdir(directory):
			try:
				mkdir(directory)
			except:
				raise IOError('Write access denied: %s'%(directory))

		wisdomFile = directory + "pyfftw_wisdom.pys"
		
		try:
			with open(wisdomFile, "wb") as f:
				pickle.dump(wisdom, f)
		except:
			raise IOError('Write access denied: %s'%(wisdomFile))

	#--------------------------------------------------------------------------
	def loadWisdom(self):
		"""Loads a former saved FFTW scheme plan from a pickle file."""
		home = expanduser("~")
		if 'win' in sys.platform:
			pyfftwDir = "\\.pyfftw\\"
		else:
			pyfftwDir = "/.pyfftw/"		

		wisdomFile = home + pyfftwDir + "pyfftw_wisdom.pys"
		try:
			with open(wisdomFile, "rb") as f:
				wisdom = pickle.load(f)
			self.setWisdom(wisdom)
		except IOError:
			# No wisdom-file present. In this case the wisdom file will
			# be generated. Therefore no error is raised here.
			pass

#==============================================================================
class FourierTransform(FourierMethodBase):
	#--------------------------------------------------------------------------
	def __init__(self, size=128, dtype='float64'):
		"""
		Returns a callable Fourier Transform object.

		size:  specify the array sizes for the input array to
		       be fourier transformed.
		dtype: specify input data type. Default is float64.
		"""

		if size < 1:
			raise ValueError("Array size must be >= 1.")
		
		self.inputArraySize = int(size)
		self.inputDataType = dtype
		self.numThreads = self.floorLog2(self.inputArraySize)

		if self.inputDataType == 'complex64':
			self.outputDataType = 'complex64'
			self.outputArraySize = self.inputArraySize
		elif self.inputDataType == 'complex128':
			self.outputDataType = 'complex128'
			self.outputArraySize = self.inputArraySize
		elif self.inputDataType == 'clongdouble':
			self.outputDataType = 'clongdouble'
			self.outputArraySize = self.inputArraySize
		elif self.inputDataType == 'float32':
			self.outputDataType = 'complex64'
			self.outputArraySize = self.inputArraySize//2 + 1
		elif self.inputDataType == 'float64':
			self.outputDataType = 'complex128'
			self.outputArraySize = self.inputArraySize//2 + 1
		#elif self.inputDataType == 'longdouble':
		#	self.outputDataType = 'clongdouble'
		#	self.outputArraySize = self.inputArraySize//2 + 1
		else:
			raise ValueError("Invalid input data type: %s"%(self.inputDataType))

		self.inputArray  = pyfftw.zeros_aligned(self.inputArraySize, dtype=self.inputDataType)
		self.outputArray = pyfftw.zeros_aligned(self.outputArraySize, dtype=self.outputDataType)

		self.loadWisdom()
		
		self.FFT = pyfftw.FFTW(self.inputArray, self.outputArray, direction='FFTW_FORWARD', threads=self.numThreads)
		self.FFT() # calling the FFT object to initiate FFTW plan for transformation

	#--------------------------------------------------------------------------
	def __del__(self):
		self.saveWisdom()

		del self.FFT
		del self.inputArray
		del self.outputArray

	#--------------------------------------------------------------------------
	def __call__(self, inputTS):
		"""Computes the fourier transform of the input TimeSeries object."""
		data = inputTS.data
		if data.ndim > 1:
			raise ValueError("Input data must be one dimentional.")
		if type(data[0]) != type(self.inputArray[0]):
			raise TypeError("Input data must be an array of type %s."%(type(self.inputArray[0])))
		if data.size != self.inputArray.size:
			raise ValueError("Input data array must have a length %E."%(self.inputArray.size))

		self.inputArray[:] = np.copy(np.array(data, dtype=self.inputDataType))
		self.FFT.update_arrays(self.inputArray, self.outputArray)
		self.FFT.execute()
		
		resultSpectrum = Spectrum(2.0*self.outputArray/self.inputArraySize, inputTS.samplFreq)
		
		return resultSpectrum
	
	#--------------------------------------------------------------------------
	def __str__(self):
		return "Fourier Transform\nInput array size:  %s\n \
	                               Input data type:   %s\n \
	                               Output array size: %s\n \
	                               Output data type:  %s"%(self.inputArraySize,\
														   self.inputDataType,\
														   self.outputArraySize,\
														   self.outputDataType)

#==============================================================================
class InverseFourierTransform(FourierMethodBase):
	#--------------------------------------------------------------------------
	def __init__(self, size=128, dtype='complex128', realOut=True):
		"""
		Returns a callable Inverse Fourier Transorm object.

		size:    specify the array sizes for the two input arrays to
		         be cross correlated.
		dtype:   specify input data type. Default float64.
		realOut: specify whether the output of cross correlation is
		         real or complex valued. Default is true.
		"""
		if size < 1:
			raise ValueError("Array size must be >= 1.")

		self.inputArraySize = int(size)
		self.inputDataType = dtype
		self.realOut = realOut
		self.numThreads = self.floorLog2(self.inputArraySize)
		
		if self.inputDataType == 'complex64' and self.realOut == True:
			self.outputDataType = 'float32'
			self.outputArraySize = (self.inputArraySize - 1) * 2
		elif self.inputDataType == 'complex128' and self.realOut == True:
			self.outputDataType = 'float64'
			self.outputArraySize = (self.inputArraySize - 1) * 2
		elif self.inputDataType == 'clongdouble' and self.realOut == True:
			self.outputDataType = 'longdouble'
			self.outputArraySize = (self.inputArraySize - 1) * 2
		elif self.inputDataType == 'complex64' and self.realOut == False:
			self.outputDataType = 'complex64'
			self.outputArraySize = self.inputArraySize
		elif self.inputDataType == 'complex128' and self.realOut == False:
			self.outputDataType = 'complex128'
			self.outputArraySize = self.inputArraySize
		#elif self.inputDataType == 'clongdouble' and self.realOut == False:
		#	self.outputDataType = 'clongdouble'
		#	self.outputArraySize = self.inputArraySize
		else:
			raise ValueError("Invalid input data type or invalid combination.")

		self.inputArray  = pyfftw.zeros_aligned(self.inputArraySize, dtype=self.inputDataType)
		self.outputArray = pyfftw.zeros_aligned(self.outputArraySize, dtype=self.outputDataType)

		self.loadWisdom()
		
		self.IFFT = pyfftw.FFTW(self.inputArray, self.outputArray, direction='FFTW_BACKWARD', threads=self.numThreads)
		self.IFFT() # calling the IFFT object to initiate FFTW plan for transformation

	#--------------------------------------------------------------------------
	def __del__(self):
		self.saveWisdom()

		del self.IFFT
		del self.inputArray
		del self.outputArray

	#--------------------------------------------------------------------------
	def __call__(self, inputSpectrum):
		"""Computes the inverse fourier transform of the input time series data(t)."""
		data = inputSpectrum.data
		if data.ndim > 1:
			raise ValueError("Input data must be one dimentional.")
		if type(data[0]) != type(self.inputArray[0]):
			raise TypeError("Input data must be an array of type %s."%(type(self.inputArray[0])))
		if data.size != self.inputArray.size:
			raise ValueError("Input data array must have a length %E."%(self.inputArray.size))
		
		self.inputArray[:] = np.copy(np.array(data, dtype=self.inputDataType))
		self.IFFT.update_arrays(self.inputArray, self.outputArray)
		self.IFFT.execute()
		
		resultTimeSeries = TimeSeries(self.outputArray/2.0, inputSpectrum.samplFreq)
		
		return resultTimeSeries

	#--------------------------------------------------------------------------
	def __str__(self):
		return "Inverse Fourier Transform\nInput array size:  %s\n \
										   Input data type:   %s\n \
										   Output array size: %s\n \
										   Output data type:  %s"%(self.inputArraySize, \
														           self.inputDataType,  \
														           self.outputArraySize,\
														           self.outputDataType)


#==============================================================================
class Convolution():
	#--------------------------------------------------------------------------
	def __init__(self, signalSize=128, responseSize=64, dtype='real64'):
		
		self.resultSize = signalSize + responseSize - 1

		self.signalSize = signalSize
		self.responseSize = responseSize
		self.dtype = dtype

		self.FFT = FourierTransform(self.resultSize, dtype)
		self.IFFT = InverseFourierTransform(size=self.FFT.outputArraySize,
			                                dtype=self.FFT.outputDataType)

	#--------------------------------------------------------------------------
	def __del__(self):
		del self.FFT
		del self.IFFT

	#--------------------------------------------------------------------------
	def __call__(self, signal):
		"""Convolves the input time series f(t) and g(t) satisfying h = f * g.
		   Here, f is the input data and g is the response function, which must
		   be set separately.

		   The used method is fast fourier transform with h(t) = IFFT(F*G),
		   where F and G are the fourier tansforms of the input time series f and g.
		   
		   IMPORTANT NOTE:
		   It is assumed, that the response function will stay the same for
		   many signals to be convolved with. Therefore, the response function
		   must be provided separately by calling setResponseFunction(response).
		   Afterwords every input signal will be convolved with this response function
		   until it is changed again by setResponseFunction(newResponse). This
		   method saves almost half of the computation time.
		"""
		
		# add zero padding to signal
		paddedSignal = TimeSeries(np.concatenate(((signal.data,
												   np.zeros(self.resultSize - self.signalSize,
														    dtype = signal.data.dtype)))),
								  fs = signal.samplFreq)
		
		if not hasattr(self, "fftResponse"):
			raise ValueError('Response function not set.')
		
		fftSignal = self.FFT(paddedSignal)
		fftResult = fftSignal * self.fftResponse
		result = self.IFFT(fftResult)
		
		return result

	#--------------------------------------------------------------------------
	def __str__(self):
		return "Convolution"
	
	#--------------------------------------------------------------------------
	@property
	def responseFunction(self):
		#TODO: return as TimeSeries or as Spectrum object...I'm not sure which one is better
		pass
	
	@responseFunction.setter
	def responseFunction(self, response):
		#TODO: edit doc string.
		"""Since the response function is assumed to be the same for many
		   signals to be convolved with, it must be initialized first to
		   be reused with every signal to be convolved.
		"""
		
		# add zero padding to responseFunction
		paddedResponse = TimeSeries(np.concatenate((response.data,
												    np.zeros(self.resultSize - self.responseSize,
												             dtype = response.data.dtype))),
									fs = response.samplFreq)
		
		self.fftResponse = self.FFT(paddedResponse)


# #==============================================================================
# class Deconvolution():
# 	#--------------------------------------------------------------------------
# 	def __init__(self, arraySize=128, dtype='float64'):
# 		self.FFT  = FourierTransform(arraySize, dtype)
# 		self.IFFT = InverseFourierTransform(arraySize, dtype)

# 	#--------------------------------------------------------------------------
# 	def __del__(self):
# 		pass

# 	#--------------------------------------------------------------------------
# 	def __call__(self, h, g):
# 		"""Deconvolves the recorded time series h(t) which was previously
# 		   convolved with a signal g(t).

# 		   Returns the function f(t), which satifies f * g = h.

# 		   The used method is fast fourier transform: f(t) = iFFT(H/G).
# 		"""
# 		#self.__checkData(h.ndim, g.ndim, h.shape, g.shape)

# 		H = self.FFT(h)
# 		G = self.FFT(g)

# 		F = np.divide(H, G)
# 		f = self.IFFT(F)

# 		return f

# 	#--------------------------------------------------------------------------
# 	def __str__(self):
# 		return "Deconvolution"

#==============================================================================
class AutoCorrelation():
	#--------------------------------------------------------------------------
	def __init__(self, size=128, dtype='float64', realOut=True):
		"""
		Returns a callable Auto-Correlation object.

		size:    specifies the array sizes for the two input arrays to
		         be cross correlated.
		dtype:   specifies input data type. Default float64.
		realOut: specifies whether the output of cross correlation is
		         real or complex data type. Default true.
		"""
		# This is implemented via Cross Correlation
		self.correlation = CrossCorrelation(size=size,
											dtype=dtype,
											realOut=realOut)

	#--------------------------------------------------------------------------
	def __del__(self):
		del self.correlation

	#--------------------------------------------------------------------------
	def __call__(self, data):
		"""Computes the auto-correlation r(tau) of the input time series x(t).

		   The method is using fast fourier transform with
		   F(f) = FFT[x(t)]
		   S(f) = F(f) * c.c.[F(f)]
		   r(tau) = IFFT[S(f)],
		   where c.c. denotes complex conjugated.
		"""
		
		return self.correlation(data, data)

	#--------------------------------------------------------------------------
	def __str__(self):
		return "Auto-Correlation\nInput array size:  %s\n \
	                              Input data type:   %s\n \
	                              Output array size: %s\n \
	                              Output data type:  %s"%(self.correlation.FFT.inputArraySize,  \
									                      self.correlation.FFT.inputDataType,   \
														  self.correlation.IFFT.outputArraySize,\
														  self.correlation.IFFT.outputDataType)


#==============================================================================
class CrossCorrelation():
	#--------------------------------------------------------------------------
	def __init__(self, size=128, dtype='float64', realOut=True,):
		"""
		Returns a callable Cross Correlation object.

		size:    specify the array sizes for the two input arrays to
		         be cross correlated.
		dtype:   specify input data type. Default float64.
		realOut: specify whether the output of cross correlation is
		         real or complex. Default true.
		"""
		if size < 1:
			raise ValueError("Input array size must be >= 1.")

		self.arraySize = size
		self.dataType = dtype
		self.correlationScale = int(2*self.arraySize)
		self.FFT  = FourierTransform(self.correlationScale, self.dataType)
		self.IFFT = InverseFourierTransform(size=self.FFT.outputArraySize,
			                                dtype=self.FFT.outputDataType,
			                                realOut=realOut)

	#--------------------------------------------------------------------------
	def __del__(self):
		del self.FFT
		del self.IFFT

	#--------------------------------------------------------------------------
	def __call__(self, tsA, tsB):
		"""Computes the cross-correlation h(tau)
		   of the input time series tsA(t) and tsB(t).

		   The method is using fast fourier transform with
		   H(f) = FFT[tsA(t)] * c.c.(FFT[tsB(t)])
		   h(tau) = IFFT[H(f)],
		   where c.c. denotes complex conjugated.
		"""
		if tsA.data.size != self.arraySize or tsB.data.size != self.arraySize:
			raise ValueError("Data arrays must both be of length %s"%(self.arraySize))
		
		paddedA = TimeSeries(np.concatenate((tsA.data,
											 np.zeros(self.correlationScale - self.arraySize,
													  dtype = self.dataType))),
							 fs = tsA.samplFreq)
		
		paddedB = TimeSeries(np.concatenate((tsB.data,
											 np.zeros(self.correlationScale - self.arraySize,
													  dtype = self.dataType))),
							 fs = tsB.samplFreq)
		
		spectrumA = self.FFT(paddedA)
		spectrumB = self.FFT(paddedB)
		fftResult = spectrumA * spectrumB.conj
		
		r = self.IFFT(fftResult)
		tmp = np.split(r.data, 2)
		result = np.roll(np.concatenate((tmp[1], tmp[0])), -1)[:-1]

		return TimeSeries(result, fs=tsA.samplFreq)

	#--------------------------------------------------------------------------
	def __str__(self):
		return "Cross-Correlation\nInput array size:  %s\n \
	                               Input data type:   %s\n \
	                               Output array size: %s\n \
	                               Output data type:  %s"%(self.FFT.inputArraySize,  \
									                       self.FFT.inputDataType,   \
														   self.IFFT.outputArraySize,\
														   self.IFFT.outputDataType)

