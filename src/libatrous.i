/* 
 * This file is part of the libatrous library
 *   (https://github.com/zindy/libatrous)
 * Copyright (c) 2017 Egor Zindy
 * 
 * libatrous is free software: you can redistribute it and/or modify  
 * it under the terms of the GNU Lesser General Public License as   
 * published by the Free Software Foundation, version 3.
 *
 * libatrous is distributed in the hope that it will be useful, but 
 * WITHOUT ANY WARRANTY; without even the implied warranty of 
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU 
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program. If not, see <http://www.gnu.org/licenses/>.
 */

%module libatrous
%{
#include <errno.h>
#include "libatrous.h"

#define SWIG_FILE_WITH_INIT
%}

%include "numpy_mod.i"
%include "exception.i"

%init %{
    import_array();
%}

%exception
{
    errno = 0;
    $action

    if (errno != 0)
    {
        switch(errno)
        {
            case EBADF:
                PyErr_Format(PyExc_ValueError, "Grid dimensions must be strictly positive");
                break;
            case EACCES:
                PyErr_Format(PyExc_ValueError, "Scale parameter must be positive");
                break;
            case E2BIG:
                PyErr_Format(PyExc_ValueError, "Image is too small (Image size must be bigger than kernel_size in all directions)");
                break;
            case ENOMEM:
                PyErr_Format(PyExc_MemoryError, "Not enough memory");
                break;
            case EPERM:
                PyErr_Format(PyExc_IndexError, "Unknown index value");
                break;
            default:
                PyErr_Format(PyExc_Exception, "Unknown exception");
        }
        SWIG_fail;
    }
}

%typemap(default) float val_z "$1=1;"

%apply (float** ARGOUTVIEW_ARRAY1, int* DIM1) {(float **kernel_ptr, int *kernel_size_ptr)}
%apply (float* IN_ARRAY3, int DIM1, int DIM2, int DIM3) {(float *ArrayIn, int Zdim, int Ydim, int Xdim)}
%apply (float* IN_ARRAY1, int DIM1) {(float *kernel, int kernel_size)}
%apply (float** ARGOUTVIEWM_ARRAY4, int* DIM1, int* DIM2, int* DIM3, int* DIM4) {(float **ArrayOut, int *SdimOut, int *ZdimOut, int *YdimOut, int *XdimOut)}
%apply (float** ARGOUTVIEWM_ARRAY3, int* DIM1, int* DIM2, int* DIM3) {(float **ArrayOut, int *ZdimOut, int *YdimOut, int *XdimOut)}
%apply (float** ARGOUTVIEWM_ARRAY3, int* DIM1, int* DIM2, int* DIM3) {(float **ArraySmooth, int *ZdimSmooth, int *YdimSmooth, int *XdimSmooth)}


%rename (scale) scale_safe;
%rename (stack) stack_safe;
%rename (iterscale) iterscale_safe;
%inline %{

void scale_safe(float *ArrayIn, int Zdim, int Ydim, int Xdim, float *kernel, int kernel_size, int the_scale, float **ArrayOut, int *ZdimOut, int *YdimOut, int *XdimOut) {
    int _Zdim = (Zdim == -1)?1:Zdim;
    int _Ydim = (Ydim == -1)?1:Ydim;
    scale(ArrayIn,_Zdim,_Ydim,Xdim,kernel,kernel_size,the_scale,ArrayOut,ZdimOut,YdimOut,XdimOut);
    *ZdimOut = Zdim;
    *YdimOut = Ydim;
}

void stack_safe(float *ArrayIn, int Zdim, int Ydim, int Xdim, float *kernel, int kernel_size, int n_scales, float **ArrayOut, int *SdimOut, int *ZdimOut, int *YdimOut, int *XdimOut) {
    int _Zdim = (Zdim == -1)?1:Zdim;
    int _Ydim = (Ydim == -1)?1:Ydim;
    stack(ArrayIn,_Zdim,_Ydim,Xdim,kernel,kernel_size,n_scales,ArrayOut,SdimOut,ZdimOut,YdimOut,XdimOut);
    *ZdimOut = Zdim;
    *YdimOut = Ydim;
}

void iterscale_safe(float *ArrayIn, int Zdim, int Ydim, int Xdim, float *kernel, int kernel_size, int the_scale, float **ArrayOut, int *ZdimOut, int *YdimOut, int *XdimOut, float **ArraySmooth, int *ZdimSmooth, int *YdimSmooth, int *XdimSmooth) {
    int _Zdim = (Zdim == -1)?1:Zdim;
    int _Ydim = (Ydim == -1)?1:Ydim;
    iterscale(ArrayIn,_Zdim,_Ydim, Xdim, kernel, kernel_size, the_scale, ArrayOut, ZdimOut, YdimOut, XdimOut, ArraySmooth, ZdimSmooth, YdimSmooth, XdimSmooth);
    *ZdimOut = Zdim;
    *YdimOut = Ydim;
    *ZdimSmooth = Zdim;
    *YdimSmooth = Ydim;
}
%}

%ignore stack;
%ignore scale;
%ignore iterscale;
%include "libatrous.h"
