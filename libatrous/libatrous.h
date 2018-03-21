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

#define LIN3 0
#define SPL5 1
#define LIN5 2
#define LG53 3
#define GAU5 4
#define CUB1 5
#define CUB2 6
#define CDF97 7
#define NUMKERN 8

int get_numkern(void);
void set_grid(float val_x, float val_y, float val_z);
void set_ncores(int val);
void enable_conv(float val_x, float val_y, float val_z);
int get_ncores(void);
void get_kernel(int kernel_index, float **kernel_ptr, int *kernel_size_ptr);
const char *get_kernel_name(int kernel_index);
void stack(float *ArrayIn, int Zdim, int Ydim, int Xdim, float *kernel, int kernel_size, int n_scales, float **ArrayOut, int *SdimOut, int *ZdimOut, int *YdimOut, int *XdimOut);
void scale(float *ArrayIn, int Zdim, int Ydim, int Xdim, float *kernel, int kernel_size, int the_scale, float **ArrayOut, int *ZdimOut, int *YdimOut, int *XdimOut);
void iterscale(float *ArrayIn, int Zdim, int Ydim, int Xdim, float *kernel, int kernel_size, int the_scale, float **ArrayOut, int *ZdimOut, int *YdimOut, int *XdimOut, float **ArraySmooth, int *ZdimSmooth, int *YdimSmooth, int *XdimSmooth);

void get_dmap(int scale, int n_scales, float sigmar, float alpha, int maxval, float **ArrayOut, int *XdimOut);
void iterscale_ea(float *ArrayIn, int Zdim, int Ydim, int Xdim, float *kernel, int kernel_size, float *dmap, int dmap_size, int the_scale, float **ArrayOut, int *ZdimOut, int *YdimOut, int *XdimOut, float **ArraySmooth, int *ZdimSmooth, int *YdimSmooth, int *XdimSmooth);

