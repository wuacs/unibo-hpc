#include "hpc.h"
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>


typedef unsigned char cell_t;

#define MAXBLKDIM 1024
#define BLKSIDE 32 /* since 32^2 = 1024 */

/* The following function makes indexing of the 2D domain
   easier. Instead of writing, e.g., grid[i*ext_width + j] you write
   IDX(grid, ext_width, i, j) to get a pointer to grid[i][j]. This
   function assumes that the size of the CA grid is
   (ext_width*ext_height), where the first and last rows/columns are
   ghost cells. */
__host__ __device__ cell_t* IDX(cell_t *grid, int ext_width, int i, int j)
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
__global__ void copy_top_bottom(cell_t *grid, int ext_width, int ext_height)
{
    const int TOP = 1;
    const int BOTTOM = ext_height - 2;
    const int TOP_GHOST = TOP - 1;
    const int BOTTOM_GHOST = BOTTOM + 1;
    const int idx = blockIdx.x * MAXBLKDIM + threadIdx.x;
    if ( idx < ext_width ) {
        *IDX(grid, ext_width, BOTTOM_GHOST, idx) = *IDX(grid, ext_width, TOP, idx); /* top to bottom halo */
        *IDX(grid, ext_width, TOP_GHOST, idx) = *IDX(grid, ext_width, BOTTOM, idx); /* bottom to top halo */
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
__global__ void copy_left_right(cell_t *grid, int ext_width, int ext_height)
{
    const int LEFT = 1;
    const int RIGHT = ext_width - 2;
    const int LEFT_GHOST = LEFT - 1;
    const int RIGHT_GHOST = RIGHT + 1;
    const int idx = blockIdx.x * MAXBLKDIM + threadIdx.x;

    if ( idx < ext_height ) {
        *IDX(grid, ext_width, idx, LEFT_GHOST) = *IDX(grid, ext_width, idx, RIGHT); /* top to bottom halo */
        *IDX(grid, ext_width, idx, RIGHT_GHOST) = *IDX(grid, ext_width, idx, LEFT); /* bottom to top halo */
    }
}

/* Compute the `next` grid given the current configuration `cur`.
   Both grids have (ext_width * ext_height) elements.
*/
__global__ void step(cell_t *cur, cell_t *next, int ext_width, int ext_height)
{
    const int LEFT = 1;
    const int RIGHT = ext_width - 2;
    const int TOP = 1;
    const int BOTTOM = ext_height - 2;
    const int idx_x = blockIdx.x * BLKSIDE + threadIdx.x + 1;
    const int idx_y = blockIdx.y * BLKSIDE + threadIdx.y + 1;
    
    if ( idx_x >= TOP && idx_x <= BOTTOM && idx_y >= LEFT && idx_y <= RIGHT) {
        int nblack = 0;
        for (int di=-1; di<=1; di++) {
            for (int dj=-1; dj<=1; dj++) {
                nblack += *IDX(cur, ext_width, idx_x+di, idx_y+dj);
            }
        }
        //printf("idx_x: %d, idx_y: %d, ci sono %d 1 intorno.\n", idx_x, idx_y, nblack);
        *IDX(next, ext_width, idx_x, idx_y) = (nblack >= 6 || nblack == 4);
        
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

    snprintf(fname, sizeof(fname), "cuda-anneal-parallel-%06d.pbm", stepno);

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
    cell_t *d_cur, *d_next;
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
    const int number_of_blocks_to_cover_x = (ext_width + MAXBLKDIM - 1)/MAXBLKDIM;
    const int number_of_blocks_to_cover_y = (ext_height + MAXBLKDIM -1)/MAXBLKDIM;
    const int number_of_blocks_to_cover_x_sided = (ext_width + BLKSIDE - 1)/BLKSIDE;
    const int number_of_blocks_to_cover_y_sided = (ext_height + BLKSIDE - 1)/BLKSIDE;

    const dim3 grid(number_of_blocks_to_cover_x_sided, number_of_blocks_to_cover_y_sided);
    const dim3 block(BLKSIDE, BLKSIDE);

    fprintf(stderr, "Anneal CA: steps=%d size=%d x %d\n", nsteps, width, height);

    cur = (cell_t*)malloc(ext_size); assert(cur != NULL);
    next = (cell_t*)malloc(ext_size); assert(next != NULL);
    init(cur, ext_width, ext_height, 0.5);
    /*for (int i=1; i<ext_height-1; i++ ) {
        for(int j=1; j<ext_width-1; j++) {
            printf("i = %d, j = %d is %d\n", i, j, *IDX(cur, ext_width, i, j));
        }
    }*/
    cudaSafeCall(cudaMalloc((void**)&d_cur, ext_size));
    cudaSafeCall(cudaMalloc((void**)&d_next, ext_size));

    cudaSafeCall(cudaMemcpy(d_cur, cur, ext_size, cudaMemcpyHostToDevice));
    const double tstart = hpc_gettime();
    for (s=0; s<nsteps; s++) {
        //printf("newstep\n");
        copy_top_bottom<<<number_of_blocks_to_cover_x, MAXBLKDIM>>>(d_cur, ext_width, ext_height);
        cudaCheckError();
        copy_left_right<<<number_of_blocks_to_cover_y, MAXBLKDIM>>>(d_cur, ext_width, ext_height);
        cudaCheckError();

#ifdef DUMPALL
        write_pbm(cur, ext_width, ext_height, s);
#endif
        step<<<grid, block>>>(d_cur, d_next, ext_width, ext_height);
        cudaCheckError();
        
        cudaMemcpy(cur, d_next, ext_size, cudaMemcpyDeviceToHost);
        cudaMemcpy(d_cur, d_next, ext_size, cudaMemcpyDeviceToDevice);
    }

    cudaFree(d_cur);
    cudaFree(d_next);
    const double elapsed = hpc_gettime() - tstart;
    write_pbm(cur, ext_width, ext_height, s);
    free(cur);
    free(next);
    fprintf(stderr, "Elapsed time: %f (%f Mops/s)\n", elapsed, (width*height/1.0e6)*nsteps/elapsed);

    return EXIT_SUCCESS;
}
