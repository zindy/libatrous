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

#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <math.h>
#include <omp.h>
 
#include "matalloc.h"
#include "libatrous.h"

tensor *convolve_3pass(tensor *tenIn, tensor *tenOut, float *kernel, int kernel_size, int the_scale);

const float k0[] = {0.25, 0.5, 0.25}; //lin3
const float k1[] = {1./16, 1./4, 3./8, 1./4, 1./16}; //bspline 5
const float k2[] = {-1./16, 0, 5./16, 0.5, 5./16, 0, -1./16}; //cubic alpha=-1
const float k3[] = {-1./32, 0, 9./32, 0.5, 9./32, 0, -1./32}; //cubic alpha=-0.5
const float k4[] = {0.0267, -0.0168, -0.0782, 0.2668, 0.6029, 0.2668, -0.0782, -0.0168, 0.0267}; //cdf97
const float k5[] = {0.06136,0.24477,0.38774,0.24477,0.06136}; //gauss 5 as per http://dev.theomader.com/gaussian-kernel-calculator/
const float k6[] = {0.125,0.25, 0.5, 0.25,0.125}; //lin 5

const float *KERNEL[7] = {k0, k1, k2, k3, k4, k5, k6};
const int KERNEL_SIZE[] = {sizeof(k0)/sizeof(float), sizeof(k1)/sizeof(float), sizeof(k2)/sizeof(float), sizeof(k3)/sizeof(float), sizeof(k4)/sizeof(float), sizeof(k5)/sizeof(float), sizeof(k6)/sizeof(float)};
const char *KERNELSTRING[] = {"Linear 3x3", "B3 Spline 5x5", "Cubic alpha=-1", "Cubic alpha=-0.5", "CDF 9/7 (JPEG 2000)", "Gaussian 5x5", "Linear 5x5"};

#define MIN(a,b) (((a)<(b))?(a):(b))
#define MAX(a,b) (((a)>(b))?(a):(b))

float x_factor=1;
float y_factor=1;
float z_factor=1;

int get_numkern(void) {
    int numkern = sizeof(KERNELSTRING)/sizeof(char *);
    return numkern;
}

void set_grid(float val_x, float val_y, float val_z)
{
    float min_val=val_x;
    min_val = MIN(val_y,min_val);
    min_val = MIN(val_z,min_val);

    if (min_val <= 0) {
        errno = EBADF; goto end;
    }

    x_factor = 1./(val_x/min_val);
    y_factor = 1./(val_y/min_val);
    z_factor = 1./(val_z/min_val);

end:
    return;
}

//return the kernel array
void get_kernel(int kernel_index, float **kernel_ptr, int *kernel_size_ptr)
{
    int numkern = sizeof(KERNELSTRING)/sizeof(char *);

    if (kernel_index < 0 || kernel_index >= numkern)
        errno = EPERM;
    else {
        *kernel_ptr = (float *)KERNEL[kernel_index];
        *kernel_size_ptr = (int)KERNEL_SIZE[kernel_index];
    }
}

//return the kernel name
const char *get_kernel_name(int kernel_index) {
    const char *ret = NULL;
    int numkern = sizeof(KERNELSTRING)/sizeof(char *);

    if (kernel_index < 0 || kernel_index >= numkern)
        errno = EPERM;
    else
        ret = KERNELSTRING[kernel_index];

    return ret;
}

//syntax uses tensor structures defined in matalloc.c.
//As we convert either 1-D, 2-D or 3-D data to tensor structures (with possibly nz=1 and ny=1), we need to be careful about how processing gets split using OpenMP.
//We want to split processing on the outerloop but for 3D, we may have less Z-planes than we have cores available.
//For 3D, it will be faster (even taking into account context switching) to split the inner loop using OpenMP.
//Then we can optimise the processing of 1-D, 2-D or 3-D data in different blocks of code.
tensor *convolve_3pass(tensor *tenIn, tensor *tenOut, float *kernel, int kernel_size, int the_scale)
{
    int x,y,z,dk,index;
    int nx,ny,nz;
    
    //the sift array depends both on the scale and the x,y,z factors (correction factor for rectangular voxels)
    int *shift_array = NULL;
    int center;

    tensor *tenTemp1= NULL, *tenTemp2 = NULL;

    float ***tarIn, ***tarOut, ***tarTemp1, ***tarTemp2;
    float dot;

    if (the_scale < 0) {
        errno = EACCES; goto end;
    }

    if (errno) goto end;

    nz = tenIn->l; ny = tenIn->m; nx = tenIn->n;
    tarIn = (float ***)tenIn->te;

    if (tenOut == NULL)
    {
        tenOut = t_get(nz,ny,nx,sizeof(float));
        if (tenOut == NULL) {
            errno = ENOMEM; goto end;
        }
    }
    else
    {
        //So we passed a ten structure to convolve_3pass, but its size does not match the input tensor.
        if (nz != tenOut->l || ny != tenOut->m || nx != tenOut->n) {
            errno = EACCES; goto end;
        }
    }
    tarOut = (float ***)tenOut->te;

    //in case 2-D or 3-D input data, otherwise, we don't actually need any addtional tensors...
    if (nz > 1 || ny > 1) {
        tenTemp1 = t_get2(nz, ny, nx, sizeof(float));
        if (errno) goto end;
        tarTemp1 = (float ***)tenTemp1->te;
    }

    //the original 3D dataset is processed in all three dimensions, starting with X
    //If nz is 1, we don't actually need to process in the Z directiom so data processed in the Y direction needs to be saved directly in the output array instead of a second temporary array.
    if (nz == 1) {
        tenTemp2 = tenOut;
        tarTemp2 = tarOut;
    } else if (ny > 1) {
        tenTemp2 = t_get2(nz, ny, nx, sizeof(float));
        if (errno) goto end;
        tarTemp2 = (float ***)tenTemp2->te;
    }

    //Create and fill the shift arrays. These are small, so creating them isn't much of a penalty even if we don't need them for all three dimensions...
    shift_array = (int *)malloc(kernel_size*sizeof(int));

    if (shift_array == NULL) {
        errno = ENOMEM; goto end;
    }

    //index of centre position in the kernel
    center = kernel_size / 2;

    //this is now the distance between adjacent pixels.
    the_scale = (int)pow(2, the_scale);


    for (dk=0;dk<kernel_size;dk++)
        shift_array[dk] = (int)roundf(the_scale*(dk-center)*x_factor);

    if (nz==1 && ny==1) {
        //Filter in the X direction
        #pragma omp parallel for        \
            default(shared) private(x,dk,index,dot)

        for (x=0; x<nx; x++) {
            dot = 0;
            for(dk=0; dk< kernel_size; dk++){
                index = x + shift_array[dk];

                if(index < 0) index = (-index) % nx;
                else if (index >=nx) index = ((2*nx-1)-index) % nx;
                if (index < 0 ) index = 0;

                dot += tarIn[0][0][index] * kernel[dk];
            }
            tarOut[0][0][x] = dot;
        }
    } else {
        //Filter in the X direction
        for (z=0;z<nz;z++) {
            #pragma omp parallel for        \
                default(shared) private(y,x,dk,index,dot)

            for (y=0; y<ny; y++) {
                for (x=0; x<nx; x++) {
                    dot = 0;
                    for(dk=0; dk<kernel_size; dk++){
                        index = x + shift_array[dk];

                        if(index < 0) index = (-index) % nx;
                        else if (index >= nx) index = ((2*nx-1)-index) % nx;
                        if (index < 0 ) index = 0;

                        dot += tarIn[z][y][index] * kernel[dk];
                    }
                    tarTemp1[z][y][x] = dot;
                }
            }
        }

        //Filter in the Y direction. When nz == 1, the output tensor is tarOut.
        for (dk=0; dk<kernel_size; dk++)
            shift_array[dk] = (int)roundf(the_scale*(dk-center)*y_factor);

        for (z=0; z<nz; z++) {
            #pragma omp parallel for        \
                default(shared) private(y,x,dk,index,dot)

            for (y=0; y<ny; y++) {
                for (x=0; x<nx; x++) {
                    dot = 0;
                    for(dk=0; dk<kernel_size; dk++) {
                        index = y + shift_array[dk];

                        if(index < 0) index = (-index) % ny;
                        else if (index >=ny) index = ((2*ny-1)-index) % ny;
                        if (index < 0 ) index = 0;

                        dot += tarTemp1[z][index][x] * kernel[dk];
                    }
                    tarTemp2[z][y][x] = dot;
                }
            }
        }

        //Filter in the Z direction if we actually have a Z stack
        if (nz > 1) {
            for (dk=0; dk<kernel_size; dk++)
                shift_array[dk] = (int)roundf(the_scale*(dk-center)*z_factor);

            for (z=0; z<nz; z++) {
                #pragma omp parallel for        \
                    default(shared) private(y,x,dk,index,dot)

                for (y=0; y<ny; y++) {
                    for (x=0; x<nx; x++) {
                    //for each slice 
                        dot = 0;
                        for(dk=0; dk<kernel_size; dk++) {
                            index = z + shift_array[dk];

                            if(index < 0) index = (-index) % nz;
                            else if (index >=nz) index = ((2*nz-1)-index) % nz;
                            if (index < 0 ) index = 0;

                            dot += tarTemp2[index][y][x] * kernel[dk];
                        }
                        tarOut[z][y][x] = dot;
                    }
                }
            }
        }
    }

end:
    if (shift_array != NULL) free(shift_array);
    t_free(tenTemp1);
    if (nz > 1) t_free(tenTemp2);
    return tenOut;
}

void stack(float *ArrayIn, int Zdim, int Ydim, int Xdim, float *kernel, int kernel_size, int n_scales, float **ArrayOut, int *SdimOut, int *ZdimOut, int *YdimOut, int *XdimOut)
{
    int i,k;

    float *decomp=NULL,*the_scale;

    //for tensor allocation
    tensor *tenIn = NULL, *tenTemp = NULL, *tenSmooth = NULL;
    float *datTemp, *datSmooth;

    //allocate output array
    if (n_scales == 0) {
        errno = EACCES; goto end;
    }

    if (*ArrayOut == NULL) {
        decomp = (float *)malloc((n_scales+1)*Zdim*Ydim*Xdim*sizeof(float));
    } else {
        decomp = *ArrayOut;
    }

    if (decomp == NULL) { errno = ENOMEM; goto end; }

    //this is what we output
    *ArrayOut = decomp;

    //allocate some tensors
    tenIn = t_wrap(ArrayIn, Zdim, Ydim, Xdim, sizeof(float));
    tenTemp = t_wrap(decomp, Zdim, Ydim, Xdim, sizeof(float));
    tenSmooth = t_get(Zdim, Ydim, Xdim, sizeof(float));
    if (errno) goto end;

    //temp = image
    t_copy(tenIn,tenTemp);

    datTemp = (float *)tenTemp->data;
    datSmooth = (float *)tenSmooth->data;

    for (k=0;k<n_scales;k++) {
        //temp filtered to smooth
        tenSmooth = convolve_3pass(tenTemp, tenSmooth, kernel, kernel_size, k);

        the_scale = decomp+(k+1)*Zdim*Ydim*Xdim;

        #pragma omp parallel for        \
            default(shared) private(i)

        for (i=0;i<Zdim*Ydim*Xdim;i++) {
            the_scale[i] = datTemp[i] - datSmooth[i];
            //temp = smooth
            datTemp[i] = datSmooth[i];
        }
    }

    //use the definition of Temp to copy the low pass at the begining of the output array.
    t_copy(tenSmooth,tenTemp);

end:
    t_free(tenIn);
    t_free(tenTemp);
    t_free(tenSmooth);

    *SdimOut = n_scales+1;
    *ZdimOut = Zdim;
    *YdimOut = Ydim;
    *XdimOut = Xdim;
    *ArrayOut = decomp;
}

void scale(float *ArrayIn, int Zdim, int Ydim, int Xdim, float *kernel, int kernel_size, int the_scale, float **ArrayOut, int *ZdimOut, int *YdimOut, int *XdimOut)
{
    int k,i;

    float *decomp=NULL;

    //for tensor allocation
    tensor *tenIn = NULL, *tenTemp = NULL, *tenSmooth = NULL;
    float *datSmooth;

    //allocate output array
    if (the_scale < 0) {
        errno = EACCES; goto end;
    }

    if (*ArrayOut == NULL) {
        decomp = (float *)malloc(Zdim*Ydim*Xdim*sizeof(float));
    } else {
        decomp = *ArrayOut;
    }
    if (decomp == NULL) { errno = ENOMEM; goto end; }

    //allocate some tensors
    //The temporary image is ArrayOut (still the case even for single scale)
    tenIn = t_wrap(ArrayIn, Zdim, Ydim, Xdim, sizeof(float));
    tenTemp = t_wrap(decomp, Zdim, Ydim, Xdim, sizeof(float));
    tenSmooth = t_get(Zdim, Ydim, Xdim, sizeof(float));
    if (errno) goto end;

    //temp = image
    t_copy(tenIn,tenTemp);
    datSmooth = (float *)tenSmooth->data;

    for (k=0;k<the_scale+1;k++) {
        //temp filtered to smooth
        tenSmooth = convolve_3pass(tenTemp, tenSmooth, kernel, kernel_size, k);

        if (k == the_scale) {
            #pragma omp parallel for        \
                default(shared) private(i)

            for (i=0;i<Zdim*Ydim*Xdim;i++) {
                decomp[i] -= datSmooth[i];
            }
            break;
        }

        //temp = smooth
        t_copy(tenSmooth,tenTemp);
    }

end:
    t_free(tenIn);
    t_free(tenTemp);
    t_free(tenSmooth);

    *ZdimOut = Zdim;
    *YdimOut = Ydim;
    *XdimOut = Xdim;
    *ArrayOut = decomp;
}

//Here the input is already a processed scale.
void iterscale(float *ArrayIn, int Zdim, int Ydim, int Xdim, float *kernel, int kernel_size, int the_scale, float **ArrayOut, int *ZdimOut, int *YdimOut, int *XdimOut, float **ArraySmooth, int *ZdimSmooth, int *YdimSmooth, int *XdimSmooth)
{
    int i;

    float *decomp=NULL;

    //for tensor allocation
    tensor *tenIn = NULL, *tenSmooth = NULL;
    float *datSmooth;

    //allocate output array
    if (the_scale < 0) {
        errno = EACCES; goto end;
    }

    if (*ArrayOut == NULL) {
        decomp = (float *)malloc(Zdim*Ydim*Xdim*sizeof(float));
    } else {
        decomp = *ArrayOut;
    }
    if (decomp == NULL) { errno = ENOMEM; goto end; }

    if (*ArraySmooth == NULL) {
        datSmooth = (float *)malloc(Zdim*Ydim*Xdim*sizeof(float));
    } else {
        datSmooth = *ArraySmooth;
    }
    if (datSmooth == NULL) { errno = ENOMEM; goto end; }

    //allocate some tensors
    //The temporary image is ArrayOut (still the case even for single scale)
    tenIn = t_wrap(ArrayIn, Zdim, Ydim, Xdim, sizeof(float));
    tenSmooth = t_wrap(datSmooth, Zdim, Ydim, Xdim, sizeof(float));
    if (errno) goto end;

    //smooth then subtract
    tenSmooth = convolve_3pass(tenIn, tenSmooth, kernel, kernel_size, the_scale);

    #pragma omp parallel for        \
        default(shared) private(i)

    for (i=0;i<Zdim*Ydim*Xdim;i++) {
        decomp[i] = ArrayIn[i] - datSmooth[i];
    }

end:
    t_free(tenIn);
    t_free(tenSmooth);

    //return the scale
    *ZdimOut = Zdim;
    *YdimOut = Ydim;
    *XdimOut = Xdim;
    *ArrayOut = decomp;

    //return the lowpass
    *ZdimSmooth = Zdim;
    *YdimSmooth = Ydim;
    *XdimSmooth = Xdim;
    *ArraySmooth = datSmooth;
}