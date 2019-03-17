import pyftlmeth.FourierMethods as fm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


#------------------------------------------------------------------------------
def createSpectrumPlot(ts, spec):
    gs = gridspec.GridSpec(3, 4)
    axTime = plt.subplot(gs[0,:])
    axMagn = plt.subplot(gs[1,:2])
    axPhas = plt.subplot(gs[-1,:2])
    axReal = plt.subplot(gs[1,2:])
    axImag = plt.subplot(gs[-1:,2:])

    axTime.plot(ts.time, ts.real)
    axTime.set_ylabel("Amplitude")
    axTime.set_xlabel("Time [sec]")
    axMagn.plot(spec.freq, spec.magnitude)
    axMagn.set_ylabel("Magnitude")
    axMagn.set_xlabel("Frequency [Hz]")
    axPhas.plot(spec.freq, spec.phase)
    axPhas.set_ylabel("Phase")
    axPhas.set_xlabel("Frequency [Hz]")
    axReal.plot(spec.freq, spec.real)
    axReal.set_ylabel("Real")
    axReal.set_xlabel("Frequency [Hz]")
    axImag.plot(spec.freq, spec.imag)
    axImag.set_ylabel("Imaginary")
    axImag.set_xlabel("Frequency [Hz]")
    plt.tight_layout()
    plt.show()

#------------------------------------------------------------------------------
def createTimeSeriesPlot(ts, spec):
    gs = gridspec.GridSpec(3, 4)
    axTime = plt.subplot(gs[0,:])
    axMagn = plt.subplot(gs[1,:2])
    axPhas = plt.subplot(gs[-1,:2])
    axReal = plt.subplot(gs[1,2:])
    axImag = plt.subplot(gs[-1:,2:])

    axTime.plot(spec.freq, spec.magnitude)
    axTime.set_ylabel("Magnitude")
    axTime.set_xlabel("Frequency [Hz]")
    axMagn.plot(ts.time, ts.magnitude)
    axMagn.set_ylabel("Magnitude")
    axMagn.set_xlabel("Time [sec]")
    axPhas.plot(ts.time, ts.phase)
    axPhas.set_ylabel("Phase")
    axPhas.set_xlabel("Time [sec]")
    axReal.plot(ts.time, ts.real)
    axReal.set_ylabel("Real")
    axReal.set_xlabel("Time [sec]")
    axImag.plot(ts.time, ts.imag)
    axImag.set_ylabel("Imaginary")
    axImag.set_xlabel("Time [sec]")
    plt.tight_layout()
    plt.show()



#==============================================================================
timeDuration = 1.0 # in seconds
timeAxis = np.linspace(0.0, timeDuration, 1000)
timeData = 1.1*np.sin(2.0*np.pi * 22.0*timeAxis)  + 2.4*np.cos(2.0*np.pi * 200*timeAxis)


# create a time series object
ts = fm.TimeSeries(timeData, timeDuration)

# now create a callable FFT object for a time series with a specific length
# and a specific input array data type.
# Note, that a created FFT object is only valid for this specific
# combination of input array size and input array data type.
# The FFT object returns an object of type Spectrum()
fft = fm.FourierTransform(timeData.size, dtype=ts.data.dtype.name)

# calculate a spectrum of the time series
spec = fft(ts)

# plot everything
createSpectrumPlot(ts, spec)


#==============================================================================
# here we do the inverse transformation

# create a callable IFFT object. Please note, that a created inverse FFT object
# is only valid for its specified data size and input data type.
# The IFFT returns an object of type TimeSeries().
ifft = fm.InverseFourierTransform(spec.data.size, dtype=spec.data.dtype.name)

# calculate time series from spectrum
tsReverse = ifft(spec)

# plot it
createTimeSeriesPlot(tsReverse, spec)


#==============================================================================
# Deleting the created FFT/IFFT objects saves the internal plans to the
# wisdom file for faster plan creation when creating new FFT objects
# with same size and data type.
del fft
del ifft
