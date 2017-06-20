/*
 * Copyright (c) 2017 Egor Zindy
 *
 * matalloc is free software: you can redistribute it and/or modify  
 * it under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <sys/types.h>
#include <time.h>
#include <errno.h>
#include <math.h>
#include <omp.h>

#include "gopt.h"
#include "matalloc.h"
#include "libatrous.h"

/* 
 * Simple program for testing atrous3d
 *
 * I used gopt for parsing the arguments.
 * http://www.purposeful.co.uk/software/gopt/
 *
 * I used elapsed as per the stackoverflow recipe:
 * http://stackoverflow.com/questions/2962785/c-using-clock-to-measure-time-in-multi-threaded-programs
 *
 * Note: in terms of pure CPU clock (timed using clock()), the thread creation increase the execution time by 10%
 *
 *
 */

extern tensor *convolve_3pass(tensor *tenIn, tensor *tenOut, float *kernel, int kernel_size, int the_scale);

int main( int argc, const char **argv )
{
    const char *arg;
    int repeats = 10;
    int kernel_index = 0;
    int nslice = 20;
    int imsize = 1000;
    int the_scale = 16;

    float *kernel;
    int kernel_size;
    
    choose_kernel(SPL5,&kernel,&kernel_size);

    clock_t c0, c1;
    double elapsed, seconds_multi=0, clock_multi=0;
    struct timespec start, finish;

    tensor *tenIn, *tenOut;

    int i;

    void *options= gopt_sort( & argc, argv, gopt_start(
        gopt_option( 'h', 0, gopt_shorts( 'h', '?' ), gopt_longs( "help", "HELP" )),
        gopt_option( 'k', GOPT_ARG, gopt_shorts( 'k' ), gopt_longs( "kernel" )),
        gopt_option( 'S', GOPT_ARG, gopt_shorts( 'S' ), gopt_longs( "size" )),
        gopt_option( 's', GOPT_ARG, gopt_shorts( 's' ), gopt_longs( "slice" )),
        gopt_option( 'n', GOPT_ARG, gopt_shorts( 'n' ), gopt_longs( "repeat" ))));

    if( gopt( options, 'h' ) )
    {
        //if any of the help options was specified
        fprintf( stdout, "This program tests the single and multi-threaded versions of the wrr algorithm\nover a blank image and reports the speed-up achieved.\n\ntest_wrr [options]\n \
    [-S #]  slice size (%dx%d by default)\n \
    [-s #]  Number of slices (%d by default)\n \
    [-k #]  kernel index (%d)\n \
    [-n #]  Number of repeat tests (%d by default)\n\n",imsize,imsize,nslice,kernel_index,repeats);

        exit( EXIT_SUCCESS );
    }

    if( gopt_arg( options, 'S', &arg)) imsize = atoi(arg);
    if( gopt_arg( options, 's', &arg)) nslice = atoi(arg);
    if( gopt_arg( options, 'k', &arg)) kernel_index = atoi(arg);
    if( gopt_arg( options, 'n', &arg)) repeats = atoi(arg);

    gopt_free( options );

    //define the blank image
    printf("Defining the in and out (%d,%d,%d) tensors...",nslice,imsize,imsize);
    tenIn = t_get(nslice,imsize,imsize,sizeof(float));
    tenOut = t_get(nslice,imsize,imsize,sizeof(float));
    printf("done!");

    if (tenIn == NULL || tenOut == NULL)
    {
        fprintf(stderr,"Not enough memory to allocate the arrays...\n");
        exit( EXIT_FAILURE );
    }

    //multi-threaded
    printf("\nTesting atrous3d convolve 3 pass (new):\n");
    seconds_multi=0; clock_multi=0;
    for (i=0; i<repeats; i++)
    {
        clock_gettime(CLOCK_MONOTONIC, &start);
        c0 = clock();

        convolve_3pass(tenIn, tenOut, kernel, kernel_size, the_scale);
        c1 = clock();
        clock_gettime(CLOCK_MONOTONIC, &finish);

        if (errno != 0)
            break;

        elapsed = (finish.tv_sec - start.tv_sec);
        elapsed += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;

        printf("  iteration %d/%d, %.3fs\n", i+1, repeats, elapsed);
        seconds_multi += elapsed;
        clock_multi += (c1-c0);
    }

    seconds_multi = seconds_multi / repeats;
    clock_multi = clock_multi / repeats;

    if (errno != 0)
    {
        fprintf(stderr,"Something went wrong...\n");
        exit( EXIT_FAILURE );
    }

    t_free(tenIn);
    t_free(tenOut);

    exit( EXIT_SUCCESS );
}
