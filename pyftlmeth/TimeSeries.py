
#==============================================================================
# TimeSeries.py
#
# Copyright (C) 2016 Stefan Lasse, lasse.stefan@gmx.de
#
# This software may be modified and distributed under the terms
# of the MIT license. See the LICENSE file for details.
#==============================================================================

from .TransformResult import TransformResult
import numpy as np

#==============================================================================
class TimeSeries(TransformResult):
    #--------------------------------------------------------------------------
    def __init__(self, data, fs=1.0):
        self._data = data
        self._fs = float(fs)    # sampling frequency in Hz
        self.__updateXaxis()

    #--------------------------------------------------------------------------
    def __str__(self):
        return "Timeseries"

    #--------------------------------------------------------------------------
    def __updateXaxis(self):
        self._xAxis = np.linspace(0.0, (self._data.size-1)//self._fs, self._data.size)

    #--------------------------------------------------------------------------
    @property
    def data(self):
        """Returns the time axis of the time series."""
        return self._data

    @property
    def time(self):
        """Returns the time axis of the time series."""
        return self._xAxis

    #--------------------------------------------------------------------------
    @property
    def samplFreq(self):
        """Returns the time duration of the time series."""
        return self._fs

    @samplFreq.setter
    def samplFreq(self, value):
        """Sets the time duration of the time series."""
        self._fs = value
        self.__updateXaxis()

    #--------------------------------------------------------------------------
    # Mathematical operations on time series
    #--------------------------------------------------------------------------
    # TODO: check for same sampling frequency or implement resampling
    #--------------------------------------------------------------------------
    def __add__(self, other):
        return TimeSeries(np.add(self._data, other.data), fs=self._fs)

    #--------------------------------------------------------------------------
    def __sub__(self, other):
        return TimeSeries(np.subtract(self._data, other.data), fs=self._fs)

    #--------------------------------------------------------------------------
    def __mul__(self, other):
        return TimeSeries(np.multiply(self._data, other.data), fs=self._fs)

    #--------------------------------------------------------------------------
    def __div__(self, other):
        pass

    #--------------------------------------------------------------------------
    @property
    def conj(self):
        return TimeSeries(np.conj(self._data), fs=self._fs)
