#! /usr/bin/env python

# System imports
from distutils.core import *
from distutils      import sysconfig

# Third-party modules - we depend on numpy for everything
import numpy

# Obtain the numpy include directory.  This logic works across numpy versions.
try:
    numpy_include = numpy.get_include()
except AttributeError:
    numpy_include = numpy.get_numpy_include()

# atrous extension module
_extension = Extension("_libatrous",
                   ["libatrous.i","matalloc.c","libatrous.c"],
                   include_dirs = [numpy_include],

                   extra_compile_args = ["--verbose","-march=native","-O3","-ftree-vectorizer-verbose=2","-ffast-math"],
                   swig_opts=['-builtin'],
                   extra_link_args=[],
                   )

_extension.extra_compile_args.append("-fopenmp")
_extension.extra_link_args.append("-lgomp")


# NumyTypemapTests setup
setup(  name        = "libatrous",
        description = "Atrous wavelet library (1d, 2d, 3d)",
        author      = "Egor Zindy",
        version     = "1.0",
        ext_modules = [_extension]
        )

