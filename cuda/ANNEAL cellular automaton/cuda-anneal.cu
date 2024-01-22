/****************************************************************************
 *
 * cuda-anneal.cu - ANNEAL cellular automaton
 *
 * Copyright (C) 2017--2022 by Moreno Marzolla <moreno.marzolla(at)unibo.it>
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
% HPC - ANNEAL cellular automaton
% Moreno Marzolla <moreno.marzolla@unibo.it>
% Last updated: 2022-11-24

The ANNEAL Callular Automaton (also known as _twisted majority rule_)
is a simple two-dimensional, binary CA defined on a grid of size $W
\times H$. Cyclic boundary conditions are assumed, so that each cell
has eight neighbors. Two cells are adjacent if they share a side or a
corner.

The automaton evolves at discrete time steps $t = 0, 1, \ldots$. The
state of a cell $x$ at time $t + 1$ depends on its state at time $t$,
and on the state of its neighbors at time $t$. Specifically, let $B_x$
be the number of cells in state 1 within the neighborhood of size $3
\times 3$ centered on $x$ (including $x$). Then, if $B_x = 4$ or $B_x
\geq 6$ the new state of $x$ is 1, otherwise the new state of $x$ is
0. See Figure 1 for some examples.

![Figure 1: Computation of the new state of the central cell of a
 block of size $3 \times 3$](cuda-anneal1.svg)

To simulate synchrnonous, concurrent updates of all cells, two domains
must be used. The state of a cell is read from the "current" domain,
and new values are written to the "next" domain. "Current" and "next"
are exchanged at the end of each step.

The initial states are chosen at random with uniform
probability. Figure 2 shows the evolution of a grid of size $256
\times 256$ after 10, 100 and 1024 steps. We observe the emergence of
"blobs" that grow over time, with the exception of small "specks".

![Figure 2: Evolution of the _ANNEAL_ CA](anneal-demo.png)

I made a short YouTube video to show the evolution of the automaton
over time:

<iframe style="display:block; margin:0 auto" width="560" height="315" src="https://www.youtube-nocookie.com/embed/TSHWSjICCxs" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

The program [cuda-anneal.cu](cuda-anneal.cu) computes the evolution of
the _ANNEAL_ CA after $K$ iterations. The final state is written to a
file. The goal of this exercise is to modify the program to use the
GPU to update the domain.

Some suggestions:

- Start writing a version that does _not_ use shared memory. Transform
  the `copy_top_bottom()`, `copy_left_right()` and `step()` functions
  into kernels. The size of the thread blocks that copy the sides of
  the domain will be different from the size of the domain that
  computes the evolution of the automaton (see the next points).

- Use a 1D array of threads to copy the ghost cells.
  `copy_top_bottom()` requires $(W + 2)$ threads, and
  `copy_left_right()` requires $(H + 2)$ threads.

- To compute new states, organize the threads in two-dimensional
  blocks of size $32 \times 32$.

- In the `step()` kernel, each thread computes the new state of a cell
  $(i, j)$. We are working with an extended domain with ghost
  rows/columns, hence the "true" (non-ghost) cells are those with
  coordinates $1 \leq i \leq H$, $1 \leq j \leq W$.  Therefore, each
  thread will compute $i, j$ as:
```C
  const int i = 1 + threadIdx.y + blockIdx.y * blockDim.y;
  const int j = 1 + threadIdx.x + blockIdx.x * blockDim.x;
```
  Before making any computation, each thread must verify that $1 \leq i
  \leq H$, $1 \leq j \leq W$, so that excess threads are
  deactivated.

## Using local memory

This program _might_ benefit from the use of shared memory, since each
cell is read 9 times by 9 different threads in the same
block. However, you might observe no improvement on modern GPUs, since
they have on-board caches. In fact, you might actually observe that
using shared memory _decreases_ the performance of the computation,
since there is not enough data reuse to compensate for the overhead of
managing the shared memory.  Despite this, it is a useful exercise to
use local memory anyway, to see how it can be done.

Let us assume that thead blocks have size $\mathit{BLKDIM} \times
\mathit{BLKDIM}$, where _BLKDIM = 32_ divides $W$ and $H$. Each block
uses a local buffer `buf[BLKDIM+2][BLKDIM+2]` which includes a ghost
area. The buffer is filled with data from the current domain, and the
new cell states are computed using the data from the buffer.

Each thread is mapped to cell $(gi, gj)$ in the global domain, and to
a copy of the same cell at coordinates $(li, lj)$ in the local
buffer. The coordinates can be computes as follows:

```C
    const int gi = 1 + threadIdx.y + blockIdx.y * blockDim.y;
    const int gj = 1 + threadIdx.x + blockIdx.x * blockDim.x;
    const int li = 1 + threadIdx.y;
    const int lj = 1 + threadIdx.x;
```

![Figure 3: Copying data from global to shared memory](cuda-anneal3.svg)

There are several ways to fill the ghost area, all of them rather
cumbersome and potentially inefficient. The solution proposed below is
one of them; other possibilities exist.

We use blocks of size $\mathit{BLKDIM} \times
\mathit{BLKDIM}$. Filling the central part of the local domain (i.e.,
everything excluding the ghost area) is done with a single instruction
executed by all threads:

```C
    buf[li][lj] = *IDX(cur, ext_width, gi, gj);
```

where `ext_width = (W + 2)` is the width of the domain including the
ghost area.

![Figure 4: Active threads while filling the shared memory](cuda-anneal4.svg)

Filling the ghost area can be done as follows (see Figure 4):

1. The upper and lower rows are delegated to the threads of the first
   row (i.e., those with $li = 1$);

2. The left and right columns are delegated to the threads of the
   first column (i.e., those with $lj = 1$);

3. The corners are delegated to the top left thread with $(li, lj) =
   (1, 1)$.

You might be tempted to collapse steps 1 and 2 into a single step that
is carried out, e.g., by the threads of the first row. This could
work, but it would be difficult to generalize the program to domains
whose sides $W, H$ are not multiple of _BLKDIM_.

In practice, you may use the following schema:

```C
    if ( li == 1 ) {
        "fill buf[0][lj] and buf[BLKDIM+1][lj]"
    }
    if ( lj == 1 ) {
        "fill buf[li][0] and buf[li][BLKDIM+1]"
    }
    if ( li == 1 && lj == 1 ) {
        "fill buf[0][0]"
        "fill buf[0][BLKDIM+1]"
        "fill buf[BLKDIM+1][0]"
        "fill buf[BLKDIM+1][BLKDIM+1]"
    }
```

Handling the ghost area is more difficult if $W, H$ are not multiple
of _BLKDIM_. Deactivating threads outside the domain is not enough:
you need to modify the code that fills the ghost area.

To compile without using shared memory:

        nvcc cuda-anneal.cu -o cuda-anneal

To generate an image after every step:

        nvcc -DDUMPALL cuda-anneal.cu -o cuda-anneal

You can make an AVI / MPEG-4 animation using:

        ffmpeg -y -i "cuda-anneal-%06d.pbm" -vcodec mpeg4 cuda-anneal.avi

To compile with shared memory:

        nvcc -DUSE_SHARED cuda-anneal.cu -o cuda-anneal-shared

To execute:

        ./cuda-anneal [steps [W [H]]]

Example:

        ./cuda-anneal 64

## References

- Tommaso Toffoli, Norman Margolus, _Cellular Automata Machines: a new
  environment for modeling_, MIT Press, 1987, ISBN 9780262526319.
  [PDF](https://people.csail.mit.edu/nhm/cam-book.pdf) from Norman
  Margolus home page.

## Files

- [cuda-anneal.cu](cuda-anneal.cu)
- [hpc.h](hpc.h)
- [Animation of the ANNEAL CA on YouTube](https://youtu.be/TSHWSjICCxs)

***/
#include "hpc.h"
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>


typedef unsigned char cell_t;

/* The following function makes indexing of the 2D domain
   easier. Instead of writing, e.g., grid[i*ext_width + j] you write
   IDX(grid, ext_width, i, j) to get a pointer to grid[i][j]. This
   function assumes that the size of the CA grid is
   (ext_width*ext_height), where the first and last rows/columns are
   ghost cells. */
cell_t* IDX(cell_t *grid, int ext_width, int i, int j)
{
    return (grid + i*ext_width + j);
}

int d_min(int a, int b)
{
    return (a<b ? a : b);
}

/*
  `grid` points to a (ext_width * ext_height) block of bytes; this
  function copies the top and bottom ext_width elements to the
  opposite halo (see figure below).

   LEFT_GHOST=0     RIGHT=ext_width-2
   | LEFT=1         | RIGHT_GHOST=ext_width-1
   | |              | |
   v v              v v
  +-+----------------+-+
  |Y|YYYYYYYYYYYYYYYY|Y| <- TOP_GHOST=0
  +-+----------------+-+
  |X|XXXXXXXXXXXXXXXX|X| <- TOP=1
  | |                | |
  | |                | |
  | |                | |
  | |                | |
  |Y|YYYYYYYYYYYYYYYY|Y| <- BOTTOM=ext_height - 2
  +-+----------------+-+
  |X|XXXXXXXXXXXXXXXX|X| <- BOTTOM_GHOST=ext_height - 1
  +-+----------------+-+

 */
/* [TODO] Transform this function into a kernel */
void copy_top_bottom(cell_t *grid, int ext_width, int ext_height)
{
    int j;
    const int TOP = 1;
    const int BOTTOM = ext_height - 2;
    const int TOP_GHOST = TOP - 1;
    const int BOTTOM_GHOST = BOTTOM + 1;

    for (j=0; j<ext_width; j++) {
        *IDX(grid, ext_width, BOTTOM_GHOST, j) = *IDX(grid, ext_width, TOP, j); /* top to bottom halo */
        *IDX(grid, ext_width, TOP_GHOST, j) = *IDX(grid, ext_width, BOTTOM, j); /* bottom to top halo */
    }
}

/*
  `grid` points to a ext_width*ext_height block of bytes; this
  function copies the left and right ext_height elements to the
  opposite halo (see figure below).

   LEFT_GHOST=0     RIGHT=ext_width-2
   | LEFT=1         | RIGHT_GHOST=ext_width-1
   | |              | |
   v v              v v
  +-+----------------+-+
  |X|Y              X|Y| <- TOP_GHOST=0
  +-+----------------+-+
  |X|Y              X|Y| <- TOP=1
  |X|Y              X|Y|
  |X|Y              X|Y|
  |X|Y              X|Y|
  |X|Y              X|Y|
  |X|Y              X|Y| <- BOTTOM=ext_height - 2
  +-+----------------+-+
  |X|Y              X|Y| <- BOTTOM_GHOST=ext_height - 1
  +-+----------------+-+

 */
/* [TODO] This function should be transformed into a kernel */
void copy_left_right(cell_t *grid, int ext_width, int ext_height)
{
    int i;
    const int LEFT = 1;
    const int RIGHT = ext_width - 2;
    const int LEFT_GHOST = LEFT - 1;
    const int RIGHT_GHOST = RIGHT + 1;

    for (i=0; i<ext_height; i++) {
        *IDX(grid, ext_width, i, RIGHT_GHOST) = *IDX(grid, ext_width, i, LEFT); /* left column to right halo */
        *IDX(grid, ext_width, i, LEFT_GHOST) = *IDX(grid, ext_width, i, RIGHT); /* right column to left halo */
    }
}

/* Compute the `next` grid given the current configuration `cur`.
   Both grids have (ext_width * ext_height) elements.

   [TODO] This function should be transformed into a kernel. */
void step(cell_t *cur, cell_t *next, int ext_width, int ext_height)
{
    const int LEFT = 1;
    const int RIGHT = ext_width - 2;
    const int TOP = 1;
    const int BOTTOM = ext_height - 2;
    for (int i=TOP; i <= BOTTOM; i++) {
        for (int j=LEFT; j <= RIGHT; j++) {
            int nblack = 0;
            for (int di=-1; di<=1; di++) {
                for (int dj=-1; dj<=1; dj++) {
                    nblack += *IDX(cur, ext_width, i+di, j+dj);
                }
            }
            *IDX(next, ext_width, i, j) = (nblack >= 6 || nblack == 4);
        }
    }
}

/* Initialize the current grid `cur` with alive cells with density
   `p`. */
void init( cell_t *cur, int ext_width, int ext_height, float p )
{
    int i, j;
    const int LEFT = 1;
    const int RIGHT = ext_width - 2;
    const int TOP = 1;
    const int BOTTOM = ext_height - 2;

    srand(1234); /* initialize PRND */
    for (i=TOP; i <= BOTTOM; i++) {
        for (j=LEFT; j <= RIGHT; j++) {
            *IDX(cur, ext_width, i, j) = (((float)rand())/RAND_MAX < p);
        }
    }
}

/* Write `cur` to a PBM (Portable Bitmap) file whose name is derived
   from the step number `stepno`. */
void write_pbm( cell_t *cur, int ext_width, int ext_height, int stepno )
{
    int i, j;
    char fname[128];
    FILE *f;
    const int LEFT = 1;
    const int RIGHT = ext_width - 2;
    const int TOP = 1;
    const int BOTTOM = ext_height - 2;

    snprintf(fname, sizeof(fname), "cuda-anneal-serial-%06d.pbm", stepno);

    if ((f = fopen(fname, "w")) == NULL) {
        fprintf(stderr, "Cannot open %s for writing\n", fname);
        exit(EXIT_FAILURE);
    }
    fprintf(f, "P1\n");
    fprintf(f, "# produced by cuda-anneal.cu\n");
    fprintf(f, "%d %d\n", ext_width-2, ext_height-2);
    for (i=LEFT; i<=RIGHT; i++) {
        for (j=TOP; j<=BOTTOM; j++) {
            fprintf(f, "%d ", *IDX(cur, ext_width, i, j));
        }
        fprintf(f, "\n");
    }
    fclose(f);
}

int main( int argc, char* argv[] )
{
    cell_t *cur, *next;
    int s, nsteps = 64, width = 512, height = 512;
    const int MAXN = 2048;

    if ( argc > 4 ) {
        fprintf(stderr, "Usage: %s [nsteps [W [H]]]\n", argv[0]);
        return EXIT_FAILURE;
    }

    if ( argc > 1 ) {
        nsteps = atoi(argv[1]);
    }

    if ( argc > 2 ) {
        width = height = atoi(argv[2]);
    }

    if ( argc > 3 ) {
        height = atoi(argv[3]);
    }

    if ( width > MAXN || height > MAXN ) { /* maximum image size is MAXN */
        fprintf(stderr, "FATAL: the maximum allowed grid size is %d\n", MAXN);
        return EXIT_FAILURE;
    }

    const int ext_width = width + 2;
    const int ext_height = height + 2;
    const size_t ext_size = ext_width * ext_height * sizeof(cell_t);

    fprintf(stderr, "Anneal CA: steps=%d size=%d x %d\n", nsteps, width, height);

    cur = (cell_t*)malloc(ext_size); assert(cur != NULL);
    next = (cell_t*)malloc(ext_size); assert(next != NULL);
    init(cur, ext_width, ext_height, 0.5);
    const double tstart = hpc_gettime();
    for (s=0; s<nsteps; s++) {
        copy_top_bottom(cur, ext_width, ext_height);
        copy_left_right(cur, ext_width, ext_height);
#ifdef DUMPALL
        write_pbm(cur, ext_width, ext_height, s);
#endif
        step(cur, next, ext_width, ext_height);
        cell_t *tmp = cur;
        cur = next;
        next = tmp;
    }
    const double elapsed = hpc_gettime() - tstart;
    write_pbm(cur, ext_width, ext_height, s);
    free(cur);
    free(next);
    fprintf(stderr, "Elapsed time: %f (%f Mops/s)\n", elapsed, (width*height/1.0e6)*nsteps/elapsed);

    return EXIT_SUCCESS;
}
