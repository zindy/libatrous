# libatrous
My "A trous" wavelet library with Python 2 bindings.

This is a library I use to band-pass filter our microscopy images. For example, speckled noise and background fluorescence can both be removed by defining an adequate band (more to come).

There are a few (hopefully useful) tricks I have implemented in the library:

* The library uses separable kernels and OpenMP for speed improvements.
* Image border artifacts are greatly reduced by using mirrored edges.
* Rectangular shaped voxels are taken into account by using re-gridding (interpolated nearest neighbour positions).

Watch this space for additional code and tutorials...

## To build the Python library on 64 bit Windows with tdm-gcc, SWIG and Anaconda:
### Things to download
* [Ananconda Python](https://www.continuum.io/downloads). This was tested with 2.7
* [SWIG](http://www.swig.org/index.php) (download swig-win)
* [Cygwin](https://www.cygwin.com/)
* [tdm64-gcc](http://tdm-gcc.tdragon.net/download)

### Package installation notes
* **tdm64-gcc:** When installing, don't forget to click "openmp" under gcc.
* **SWIG:** Make a note where you unzip the swigwin folder (e.g. a folder under C:\TDM-GCC-64)
* **Anaconda:** Global install, make sure %PYTHONPATH% is defined.
* **Cygwin:** A basic install will do, no need to install gcc

### Additional steps

In a command window, type the following to create a libpython27.a usable by gcc:
```
"C:\TDM-GCC-64\x86_64-w64-mingw32\bin\gendef.exe" %PYTHONPATH%\python27.dll
"C:\TDM-GCC-64\bin\dlltool.exe" --as-flags=--64 -m i386:x86-64 -k -l libpython27.a -d python27.def
move libpython27.a C:\TDM-GCC-64\x86_64-w64-mingw32\lib
move "%PYTHONPATH%\libs\python27.lib" "%PYTHONPATH%\libs\python27.lib____"
```
You will need to modify the \Anaconda2\Lib\distutils\cygwincompiler.py as per [this stackoverflow solution](https://stackoverflow.com/questions/6034390/compiling-with-cython-and-mingw-produces-gcc-error-unrecognized-command-line-o):

  4) In the same module, modify get_msvcr() to return an empty list instead of ['msvcr90'] when msc_ver == '1500' .

Then launch cygwin terminal and add a line to your .bashrc to add the Anaconda Python, tdm64-gcc and SWIG paths:

`export PATH=/cygdrive/c/Anaconda2:/cygdrive/c/TDM-GCC-64/swigwin-3.0.11:/cygdrive/c/TDM-GCC-64/bin:$PATH`

Relaunch the cygwin terminal and these should all return values:

```
gcc -v
swig -version
python --version
```

### In place compilation:
```
cd src
python setup.py build_ext --inplace
```

## In place compilation in Linux:

Make sure you have python, gcc and swig installed. You will need to change the src/setup.cfg to
```
[build]
compiler=gcc

[build_ext]
inplace=1
```

Then it's the same
```
cd src
python setup.py build_ext --inplace
```

## Finally...
Finally, you can copy the ```libatrous.py``` and ```_libatrous.pyd``` or ```_libatrous.so``` where you need them.
For example, you can copy these in the tests folder and try ```test_libatrous.py```

I think this covers about all the steps. If you have a simpler workflow, I would of course be happy to hear about it and amend these instructions.
