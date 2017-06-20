# libatrous
My "A trous" wavelet library with Python 2 bindings.

This is a library I use to band-pass filter our microscopy images. For example, speckled noise and background fluorescence can both be removed by defining an adequate band (more to come).

There are a few (hopefully useful) tricks I have implemented in the library:

* The library uses separable kernels and OpenMP for speed improvements.
* Image border artifacts are greatly reduced by using mirrored edges.
* Rectangular shaped voxels are taken into account by using re-gridding (interpolated nearest neighbour positions).

Watch this space for additional code and tutorials...

## To build the Python library:
I've only really tested this with (tdm-gcc) MinGW and gcc in a Linux distribution. I'll add any steps to make this work with Anaconda Python when I try it on a clean install.

```
cd src
python setup.py build_ext --inplace
```
Then you can copy the ```libatrous.py``` and ```_libatrous.pyd``` or ```_libatrous.so``` where you need them.

For example, you can copy these in the tests folder and try ```test_libatrous.py```
