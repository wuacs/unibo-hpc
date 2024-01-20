/****************************************************************************
 *
 * cuda-dot.cu - Dot product
 *
 * Copyright (C) 2017--2023 by Moreno Marzolla <moreno.marzolla(at)unibo.it>
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 ****************************************************************************/

/***
% HPC - Dot product
% Moreno Marzolla <moreno.marzolla@unibo.it>
% Last updated: 2023-11-29

## Familiarize with the environment

The server has three identical GPUs (NVidia GeForce GTX 1070). The
first one is used by default, although it is possible to select
another GPU either programmatically (`cudaSetDevice(0)` uses the
first GPU, `cudaSetDevice(1)` uses the second one, and so on), or
using the environment variable `CUDA_VISIBLE_DEVICES`.

For example

        CUDA_VISIBLE_DEVICES=0 ./cuda-stencil1d

runs `cuda-stencil1d` on the first GPU (default), while

        CUDA_VISIBLE_DEVICES=1 ./cuda-stencil1d

runs the program on the second GPU.

Run `deviceQuery` from the command line to display the hardware
features of the GPUs.

## Scalar product

The program [cuda-dot.cu](cuda-dot.cu) computes the dot product of two
arrays `x[]` and `y[]` of length $n$. Modify the program to use the
GPU, by defining a suitable kernel and modifying the `dot()` function
to use it. The dot product $s$ of two arrays `x[]` and `y[]` is defined as

$$
s = \sum_{i=0}^{n-1} x[i] \times y[i]
$$

In this exercise we implement a simple (although not efficient)
approach where we use a _single_ block of _BLKDIM_ threads.  The
algorithm works as follows:

1. The CPU allocates a float array `d_tmp[]` of length _BLKDIM_ on the GPU,
   in addition to a copy of `x[]` and `y[]`.

2. The CPU executes a single 1D thread block containing _BLKDIM_
   threads; use the maximum number of threads per block supported by
   the hardware, which is _BLKDIM = 1024_.

3. Thread $t$ ($t = 0, \ldots, \mathit{BLKDIM}-1$) computes the value
   of the expression $(x[t] \times y[t] + x[t + \mathit{BLKDIM}]
   \times y[t + \mathit{BLKDIM}] + x[t + 2 \times \mathit{BLKDIM}]
   \times y[t + 2 \times \mathit{BLKDIM}] + \ldots)$ and stores the
   result in `d_tmp[t]` (see Figure 1).

4. When the kernel terminates, the CPU transfers `d_tmp[]` back to host
   memory and performs a sum-reduction to compute the final result.

![Figure 1](cuda-dot.svg)

Your program must work correctly for any value of $n$, even if it is
not a multiple of _BLKDIM_.

A better way to compute a reduction will be shown in future lectures.

To compile:

        nvcc cuda-dot.cu -o cuda-dot -lm

To execute:

        ./cuda-dot [len]

Example:

        ./cuda-dot

## Files

- [cuda-dot.cu](cuda-dot.cu)
- [hpc.h](hpc.h)

***/
#include "hpc.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>


float dot( const float *x, const float *y, int n )
{
    /* [TODO] modify this function so that (part of) the computation
       is executed on the GPU. You may want to follow the steps
       below. */

    /* Define a `float` array tmp[] of size BLKDIM on host memory */

    /* Define pointers to copies of x, y and tmp in device memory */

    /* Allocate space for device copies of x,y */

    /* Copy x, y from host to device */

    /* Launch a suitable kernel on the GPU */

    /* Copy the result back to host memory */

    /* Perform the final reduction on the CPU */

    /* Free device memory */

    float result = 0.0;
    for (int i = 0; i < n; i++) {
        result += x[i] * y[i];
    }
    return result;
}

void vec_init( float *x, float *y, int n )
{
    int i;
    const float tx[] = {1.0/64.0, 1.0/128.0, 1.0/256.0};
    const float ty[] = {1.0, 2.0, 4.0};
    const size_t LEN = sizeof(tx)/sizeof(tx[0]);

    for (i=0; i<n; i++) {
        x[i] = tx[i % LEN];
        y[i] = ty[i % LEN];
    }
}

int main( int argc, char* argv[] )
{
    float *x, *y, result;
    int n = 1024*1024;
    const int MAX_N = 128 * n;

    if ( argc > 2 ) {
        fprintf(stderr, "Usage: %s [len]\n", argv[0]);
        return EXIT_FAILURE;
    }

    if ( argc > 1 ) {
        n = atoi(argv[1]);
    }

    if ( n > MAX_N ) {
        fprintf(stderr, "FATAL: the maximum length is %d\n", MAX_N);
        return EXIT_FAILURE;
    }

    const size_t SIZE = n*sizeof(*x);

    /* Allocate space for host copies of x, y */
    x = (float*)malloc(SIZE);
    assert(x != NULL);
    y = (float*)malloc(SIZE);
    assert(y != NULL);
    vec_init(x, y, n);
    const float expected = ((float)n)/64;

    printf("Computing the dot product of %d elements...\n", n);
    result = dot(x, y, n);
    printf("got=%f, expected=%f\n", result, expected);

    /* Check result */
    if ( fabs(result - expected) < 1e-5 ) {
        printf("Check OK\n");
    } else {
        printf("Check FAILED\n");
    }

    /* Cleanup */
    free(x);
    free(y);

    return EXIT_SUCCESS;
}
