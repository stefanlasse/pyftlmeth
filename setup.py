from distutils.core import setup
setup(
  name = 'pyftlmeth',
  packages = ['pyftlmeth'], # this must be the same as the name above
  version = '0.1',
  description = 'A python library for fast Fourier transformational methods based on FFTW and pyfftw.',
  author = 'Stefan Lasse',
  author_email = 'lasse.stefan@gmx.de',
  url = 'https://github.com/stefanlasse/pyftlmeth', # use the URL to the github repo
  download_url = 'https://github.com/stefanlasse/pyftlmeth',
  keywords = ['fourier', 'transform', 'correlation', 'cross correlation', 'auto correlation', 
'convolution', 'deconvolution'], # arbitrary keywords
  classifiers = [],
)
