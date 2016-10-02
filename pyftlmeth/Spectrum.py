
#==============================================================================
# Spectrum.py
#
# Copyright (C) 2016 Stefan Lasse, lasse.stefan@gmx.de
#
# This software may be modified and distributed under the terms
# of the MIT license. See the LICENSE file for details.
#============================================================================== 

from TransformResult import TransformResult
import numpy as np

#==============================================================================
class Spectrum(TransformResult):
	#--------------------------------------------------------------------------
	def __init__(self, data, fs=1.0):
		self._data = data
		self._fs = fs
		self.__updateXaxis()
	
	#--------------------------------------------------------------------------
	def __str__(self):
		return "Spectrum"
	
	#--------------------------------------------------------------------------
	def __updateXaxis(self):
		self._xAxis = np.linspace(0.0, self._fs/2.0, self._data.size)
	
	#--------------------------------------------------------------------------
	@property
	def freq(self):
		"""Returns the spectrum's frequency axis."""
		return self._xAxis

	#--------------------------------------------------------------------------
	@property
	def samplFreq(self):
		"""Returns the spectrum's sampling frequency."""
		return self._fs

	@samplFreq.setter
	def samplFreq(self, value):
		"""Sets the spectrum's sampling frequency."""
		self._fs = value
		self.__updateXaxis()

	#--------------------------------------------------------------------------
	# Mathematical operations on spectra
	#--------------------------------------------------------------------------
	# TODO: check for same sampling frequency or implement resampling
	#--------------------------------------------------------------------------
	def __add__(self, other):
		return Spectrum(np.add(self._data, other.data), fs=self._fs)
	
	#--------------------------------------------------------------------------
	def __sub__(self, other):
		return Spectrum(np.subtract(self._data, other.data), fs=self._fs)
	
	#--------------------------------------------------------------------------
	def __mul__(self, other):
		return Spectrum(np.multiply(self._data, other.data), fs=self._fs)

	#--------------------------------------------------------------------------
	def __div__(self, other):
		pass

	#--------------------------------------------------------------------------
	@property
	def conj(self):
		return Spectrum(np.conj(self._data), fs=self._fs)

