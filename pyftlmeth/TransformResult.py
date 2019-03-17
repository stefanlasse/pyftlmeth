#==============================================================================
# TransormResult.py
#
# Copyright (C) 2016 Stefan Lasse, lasse.stefan@gmx.de
#
# This software may be modified and distributed under the terms
# of the MIT license. See the LICENSE file for details.
#==============================================================================

import numpy as np

#==============================================================================
class TransformResult():
    #--------------------------------------------------------------------------
    # def __init__(self, value):
        #if value.dtype not in np.sctypes['complex']:
        #    raise TypeError("%s must be of type complex."%(self.__str__()))
        #self._data = np.copy(value)
        # pass

    #--------------------------------------------------------------------------
    # def __del__(self):
        # del self._data
        # del self._xAxis

    #--------------------------------------------------------------------------
    @property
    def magnitude(self):
        """Returns the magnitude of the data."""
        return np.abs(self._data)

    #--------------------------------------------------------------------------
    @property
    def phase(self):
        """Returns the phase of the data."""
        return np.arctan2(np.real(self._data), np.imag(self._data))

    #--------------------------------------------------------------------------
    @property
    def real(self):
        """Returns the real part of the data."""
        return np.real(self._data)

    #--------------------------------------------------------------------------
    @property
    def imag(self):
        """Returns the imaginary part of the data."""
        return np.imag(self._data)

    #--------------------------------------------------------------------------
    @property
    def data(self):
        """Returns the result data. This can be real or complex."""
        return self._data

    @data.setter
    def data(self, value):
        """Sets a complex data array."""
        #if value.dtype not in np.sctypes['complex']:
        #    raise TypeError("Data must be of type complex.")

        self._data = np.copy(value)

    #--------------------------------------------------------------------------
    def setMagnitudePhase(self, magnitude, phase):
        """
        Sets the internal data by magnitude and phase.
        magnitude and phase must be one-dimensional and of same size.

        Parameters
        ----------
        magnitude : array_like, one-dimensional
                    The data-type of magnitude must be float.
        phase : array_like, one-dimensional
                The data-type of phase must be float.
        """
        if real.size != imag.size:
            raise ValueError("Magnitude and phase part must be of same size.")

        complexDataType = self._checkDataTypes(magnitude.dtype, phase.dtype)

        real = np.multiply(np.cos(phase), magnitude)
        imag = np.multiply(np.sin(phase), magnitude)

        self.setRealImag(real, imag)

    #--------------------------------------------------------------------------
    def setRealImag(self, real, imag):
        """
        Sets the internal data by real and imaginary parts.
        real and imag must be one-dimensional and of same size.

        Parameters
        ----------
        real : array_like, one-dimensional
               The data type of real must be float.
        imag : array_like, one-dimensional
               The data type of imag must be float.
        """
        if real.size != imag.size:
            raise ValueError("Real and imaginary part must be of same size.")

        complexDataType = self._checkDataTypes(real.dtype, imag.dtype)
        self._data = np.empty(real.size, dtype=complexDataType)
        self._data[:] = real + 1j*imag

    #--------------------------------------------------------------------------
    def _checkDataTypes(self, dtypeA, dtypeB):
        if dtypeA not in np.sctypes['float']:
            raise ValueError("Real data must be of type float.")

        if dtypeB not in np.sctypes['float']:
            raise ValueError("Imaginary data must be of type float.")

        if dtypeA != dtypeB:
            raise ValueError("Real and imaginary part must be of same dtype.")

        if dtypeA == np.float32:
            complexDataType = np.complex64
        elif dtypeA == np.float64:
            complexDataType = np.complex128
        else:
            raise ValueError("Invalid input data type or invalid combination.")

        return complexDataType
