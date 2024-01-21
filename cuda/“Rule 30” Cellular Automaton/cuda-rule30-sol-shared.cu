#include "hpc.h"
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#define BLOCKSIZE 1024

typedef unsigned char cell_t;

/**
 * Given the current state of the CA, compute the next state.  This
 * version requires that the `cur` and `next` arrays are extended with
 * ghost cells; therefore, `ext_n` is the length of `cur` and `next`
 * _including_ ghost cells.
 *
 *                             +----- ext_n-2
 *                             |   +- ext_n-1
 *   0   1                     V   V
 * +---+-------------------------+---+
 * |///|                         |///|
 * +---+-------------------------+---+
 *
 */

__global__ void kernel_step_with_sharing( cell_t *cur, cell_t *next, int n )
{
    int real_idx = threadIdx.x + blockIdx.x * BLOCKSIZE + 1; 
    if ( real_idx < n - 1 ) {
        __shared__ cell_t block[BLOCKSIZE+2];
        int shared_blk_idx = threadIdx.x + 1;
        if (threadIdx.x == 0) {
            block[0] = cur[real_idx-1];
        } else if (threadIdx.x == BLOCKSIZE-1) {
            block[BLOCKSIZE+1] = cur[real_idx+1];
        } else if (real_idx == n - 2) {
            block[shared_blk_idx+1] = cur[real_idx+1];
        }
        block[shared_blk_idx] = cur[real_idx];

        __syncthreads();
        
        const cell_t left   = block[shared_blk_idx-1];
        const cell_t center = block[shared_blk_idx  ];
        const cell_t right  = block[shared_blk_idx+1];
        next[real_idx] =
                ( left && !center && !right) ||
                (!left && !center &&  right) ||
                (!left &&  center && !right) ||
                (!left &&  center &&  right);
    }
}


/**
 * Initialize the domain; all cells are 0, with the exception of a
 * single cell in the middle of the domain. `cur` points to an array
 * of length `ext_n`; the length includes two ghost cells.
 */
void init_domain( cell_t *cur, int ext_n )
{
    int i;
    for (i=0; i<ext_n; i++) {
        cur[i] = 0;
    }
    cur[ext_n/2] = 1;
}

/**
 * Dump the current state of the CA to PBM file `out`. `cur` points to
 * an array of length `ext_n` that includes two ghost cells.
 */
void dump_state( FILE *out, const cell_t *cur, int ext_n )
{
    int i;
    const int LEFT = 1;
    const int RIGHT = ext_n - 2;
    for (i=LEFT; i<=RIGHT; i++) {
        fprintf(out, "%d ", cur[i]);
    }
    fprintf(out, "\n");
}

int main( int argc, char* argv[] )
{
    const char *outname = "cuda-rule30-shared.pbm";
    FILE *out;
    int width = 1024, steps = 1024, s;
    cell_t *cur, *next;
    cell_t *d_cur, *d_next;

    if ( argc > 3 ) {
        fprintf(stderr, "Usage: %s [width [steps]]\n", argv[0]);
        return EXIT_FAILURE;
    }

    if ( argc > 1 ) {
        width = atoi(argv[1]);
    }

    if ( argc > 2 ) {
        steps = atoi(argv[2]);
    }

    const int ext_width = width + 2;
    const size_t ext_size = ext_width * sizeof(*cur); /* includes ghost cells */
    const size_t size = width * sizeof(*cur);
    const int LEFT_GHOST = 0;
    const int LEFT = 1;
    const int RIGHT_GHOST = ext_width - 1;
    const int RIGHT = RIGHT_GHOST - 1;
    /* Create the output file */
    out = fopen(outname, "w");
    if ( !out ) {
        fprintf(stderr, "FATAL: cannot create file \"%s\"\n", outname);
        return EXIT_FAILURE;
    }
    fprintf(out, "P1\n");
    fprintf(out, "# produced by cuda-rule30-sol-shared.cu\n");
    fprintf(out, "%d %d\n", width, steps);

    /* Allocate space for the `cur[]` and `next[]` arrays */
    cur = (cell_t*)malloc(ext_size); assert(cur != NULL);
    next = (cell_t*)malloc(ext_size); assert(next != NULL);

    /* Initialize the domain */
    init_domain(cur, ext_width);

    cudaSafeCall( cudaMalloc((void**)&d_cur, ext_size ) );
    cudaSafeCall( cudaMalloc((void**)&d_next, ext_size) );

    cudaSafeCall( cudaMemcpy(d_cur, cur, ext_size, cudaMemcpyHostToDevice) );
    
    /* Evolve the CA */
    for (s=0; s<steps; s++) {

        /* Dump the current state */
        dump_state(out, cur, ext_width);

        /* Fill ghost cells */
        cudaSafeCall( cudaMemcpy(&d_cur[RIGHT_GHOST], &cur[LEFT], sizeof(cell_t), cudaMemcpyHostToDevice) );
        cudaSafeCall( cudaMemcpy(&d_cur[LEFT_GHOST], &cur[RIGHT], sizeof(cell_t), cudaMemcpyHostToDevice) );

        /* Compute next state */
        kernel_step_with_sharing
                    <<<(width + BLOCKSIZE - 2)/BLOCKSIZE, BLOCKSIZE>>>
                    (d_cur, d_next, ext_width); /*  Here we split width - 1 in blocks
                                                so last thread of last block
                                                who is going to do a
                                                computation(i.e compute next[n-2]) will do
                                                idx = idx + 1 and thus work on
                                                next[n-2] instead of next[n-3]. */
                                                                                        
        cudaCheckError();
        /* put next into cur */
        cudaSafeCall( cudaMemcpy(&d_cur[1], &d_next[1], size, cudaMemcpyDeviceToDevice) );
        cudaSafeCall( cudaMemcpy(cur, d_cur, ext_size, cudaMemcpyDeviceToHost) );
    }

    cudaFree(d_cur);
    cudaFree(d_next);
    free(cur);
    free(next);

    fclose(out);

    return EXIT_SUCCESS;
}
