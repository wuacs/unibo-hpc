/****************************************************************************
 *
 * mpi-mandelbrot.c - Mandelbrot set
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
% HPC - Mandelbrot set
% Moreno Marzolla <moreno.marzolla@unibo.it>
% Last updated: 2023-11-13

![](mandelbrot-set.png)

The file [mpi-mandelbrot.c](mpi-mandelbrot.c) contains a MPI program
that computes the Mandelbrot set; it is not a parallel program,
because the master process does everything.

The program accepts the image height as an optional command-line
parameter; the width is computed automatically to include the whole
set. Process 0 writes the output image to the file `mandebrot.ppm` in
PPM (_Portable Pixmap_) format. To convert the image, e.g., to PNG you
can use the following command on the Linux server:

        convert mandelbrot.ppm mandelbrot.png

Write a parallel version where all MPI processes contribute to the
computation. To do this, we can partition the image into $P$ vertical
blocks where $P$ is the number of MPI processes, and let each process
draws a portion of the image (see Figure 1).

![Figure 1: Domain decomposition for the computation of the Mandelbrot
 set with 4 MPI processes](mpi-mandelbrot.png)

Specifically, each process computes a portion of the image of size
$\mathit{xsize} \times (\mathit{ysize} / P)$ (see below how to handle
the case where _ysize_ is not an integer multiple of $P$). This is an
_embarrassingly parallel_ computation, since there is no need to
communicate. At the end, the processes send their local result to the
master using the `MPI_Gather()` function, so that the master can
assemble the image. We use three bytes to encode the color of each
pixel, so the `MPI_Gather()` operation will transfer blocks of $(3
\times \mathit{xsize} \times \mathit{ysize} / P)$ elements of type
`MPI_BYTE`.

You can initially assume that _ysize_ is an integer multiple of $P$,
and then relax this assumption, e.g., by letting process 0 take care
of the last `(ysize % P)` rows. Alternatively, you can use blocks of
different sizes and use `MPI_Gatherv()` to combine them.

You might want to keep the serial program as a reference. To check the
correctness of the parallel implementation, compare the output images
produced by the serial and parallel versions with the command:

        cmp file1 file2

They must be identical, i.e., the `cmp` program should print no
message.

To compile:

        mpicc -std=c99 -Wall -Wpedantic mpi-mandelbrot.c -o mpi-mandelbrot

To execute:

        mpirun -n NPROC ./mpi-mandelbrot [ysize]

Example:

        mpirun -n 4 ./mpi-mandelbrot 800

## Files

- [mpi-mandelbrot.c](mpi-mandelbrot.c)

***/
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <assert.h>
#include <mpi.h>

const int MAXIT = 100;

/* The __attribute__(( ... )) definition is gcc-specific, and tells
   the compiler that the fields of this structure should not be padded
   or aligned in any way. */
typedef struct __attribute__((__packed__)) {
    uint8_t r;  /* red   */
    uint8_t g;  /* green */
    uint8_t b;  /* blue  */
} pixel_t;

/* color gradient from https://stackoverflow.com/questions/16500656/which-color-gradient-is-used-to-color-mandelbrot-in-wikipedia */
const pixel_t colors[] = {
    {66, 30, 15}, /* r, g, b */
    {25, 7, 26},
    {9, 1, 47},
    {4, 4, 73},
    {0, 7, 100},
    {12, 44, 138},
    {24, 82, 177},
    {57, 125, 209},
    {134, 181, 229},
    {211, 236, 248},
    {241, 233, 191},
    {248, 201, 95},
    {255, 170, 0},
    {204, 128, 0},
    {153, 87, 0},
    {106, 52, 3} };
const int NCOLORS = sizeof(colors)/sizeof(colors[0]);

/*
 * Iterate the recurrence:
 *
 * z_0 = 0;
 * z_{n+1} = z_n^2 + cx + i*cy;
 *
 * Returns the first `n` such that `z_n > bound`, or `MAXIT` if `z_n` is below
 * `bound` after `MAXIT` iterations.
 */
int iterate( float cx, float cy )
{
    float x = 0.0f, y = 0.0f, xnew, ynew;
    int it;
    for ( it = 0; (it < MAXIT) && (x*x + y*y <= 2.0*2.0); it++ ) {
        xnew = x*x - y*y + cx;
        ynew = 2.0*x*y + cy;
        x = xnew;
        y = ynew;
    }
    return it;
}

/* Draw the rows of the Mandelbrot set from `ystart` (inclusive) to
   `yend` (excluded) to the bitmap pointed to by `p`. Note that `p`
   must point to the beginning of the bitmap where the portion of
   image will be stored; in other words, this function writes to
   pixels p[0], p[1], ... `xsize` and `ysize` MUST be the sizes
   of the WHOLE image. */
void draw_lines( int ystart, int yend, pixel_t* p, int xsize, int ysize )
{
    int x, y;
    for ( y = ystart; y < yend; y++) {
        for ( x = 0; x < xsize; x++ ) {
            const float cx = -2.5 + 3.5 * (float)x / (xsize - 1);
            const float cy = 1 - 2.0 * (float)y / (ysize - 1);
            const int v = iterate(cx, cy);
            if (v < MAXIT) {
                p->r = colors[v % NCOLORS].r;
                p->g = colors[v % NCOLORS].g;
                p->b = colors[v % NCOLORS].b;
            } else {
                p->r = p->g = p->b = 0;
            }
            p++;
        }
    }
}

int main( int argc, char *argv[] )
{
    int my_rank, comm_sz;
    FILE *out = NULL;
    const char* fname="mpi-mandelbrot-serial.ppm";
    pixel_t *bitmap = NULL;
    int xsize, ysize;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);

    if ( argc > 1 ) {
        ysize = atoi(argv[1]);
    } else {
        ysize = 1024;
    }

    xsize = ysize * 1.4;

    /* xsize and ysize are known to all processes */
    if ( 0 == my_rank ) {
        out = fopen(fname, "w");
        if ( !out ) {
            fprintf(stderr, "Error: cannot create %s\n", fname);
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }

        /* Write the header of the output file */
        fprintf(out, "P6\n");
        fprintf(out, "%d %d\n", xsize, ysize);
        fprintf(out, "255\n");

        /* Allocate the complete bitmap */
        bitmap = (pixel_t*)malloc(xsize*ysize*sizeof(*bitmap));
        assert(bitmap != NULL);
        /* [TODO] This is not a true parallel version, since the master
           does everything */
        draw_lines(0, ysize, bitmap, xsize, ysize);
        fwrite(bitmap, sizeof(*bitmap), xsize*ysize, out);
        fclose(out);
        free(bitmap);
    }


    MPI_Finalize();

    return EXIT_SUCCESS;
}
