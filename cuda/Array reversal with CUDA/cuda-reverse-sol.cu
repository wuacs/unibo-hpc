#include "hpc.h"
#include <stdio.h>
#include <math.h>
#include <assert.h>

#define BLKLEN 1024 /* May want to modify this */

__global__ void ker_reverse(int *in, int *out, int n) {
    int pos_in = threadIdx.x + blockIdx.x * BLKLEN;
    int pos_out = n - pos_in - 1;
    if (pos_in < n) {
        out[pos_out] = in[pos_in];
    }
}

__global__ void ker_inplace_reverse(int *in, int n) {
    int pos_in = threadIdx.x + blockIdx.x * BLKLEN;
    int pos_out = n - pos_in - 1;
    if (pos_in < n/2) {
        int temp = in[pos_out];  
        in[pos_out] = in[pos_in];
        in[pos_in] = temp;
    }
}

/* Reverses `in[]` into `out[]`; assumes that `in[]` and `out[]` do not
   overlap.
 */
void reverse( int *in, int *out, int n )
{
    int *d_in, *d_out;
    const size_t size = n * sizeof(int);
    
    cudaMalloc((void**)&d_in, size);
    cudaMalloc((void**)&d_out, size);

    cudaMemcpy( d_in, in, size, cudaMemcpyHostToDevice );
    cudaMemcpy( d_out, out, size, cudaMemcpyHostToDevice );
    
    ker_reverse<<<(n + BLKLEN - 1)/BLKLEN, BLKLEN>>>(d_in, d_out, n);

    cudaMemcpy( out, d_out, size, cudaMemcpyDeviceToHost );

    cudaFree(d_in);
    cudaFree(d_out);

}


/* In-place reversal of in[] into itself. */
void inplace_reverse( int *in, int n )
{
    int *d_in;
    const size_t size = n * sizeof(int);
    
    cudaMalloc((void**)&d_in, size);

    cudaMemcpy( d_in, in, size, cudaMemcpyHostToDevice );
    
    ker_inplace_reverse<<<(n + BLKLEN - 1)/BLKLEN, BLKLEN>>>(d_in, n);

    cudaMemcpy( in, d_in, size, cudaMemcpyDeviceToHost );

    cudaFree(d_in);
    
}

void fill( int *x, int n )
{
    int i;
    for (i=0; i<n; i++) {
        x[i] = i;
    }
}

int check( const int *x, int n )
{
    int i;
    for (i=0; i<n; i++) {
        if (x[i] != n - 1 - i) {
            fprintf(stderr, "Test FAILED: x[%d]=%d, expected %d\n", i, x[i], n-1-i);
            return 0;
        }
    }
    printf("Test OK\n");
    return 1;
}

int main( int argc, char* argv[] )
{
    int *in, *out;
    int n = 1024*1024;
    const int MAX_N = 512*1024*1024;

    if ( argc > 2 ) {
        fprintf(stderr, "Usage: %s [n]\n", argv[0]);
        return EXIT_FAILURE;
    }

    if ( argc > 1 ) {
        n = atoi(argv[1]);
    }

    if ( n > MAX_N ) {
        fprintf(stderr, "FATAL: input too large (maximum allowed length is %d)\n", MAX_N);
        return EXIT_FAILURE;
    }

    const size_t SIZE = n * sizeof(*in);

    /* Allocate in[] and out[] */
    in = (int*)malloc(SIZE);
    assert(in != NULL);
    out = (int*)malloc(SIZE);
    assert(out != NULL);
    fill(in, n);

    printf("Reverse %d elements... ", n);
    reverse(in, out, n);
    check(out, n);

    printf("In-place reverse %d elements... ", n);
    inplace_reverse(in, n);
    check(in, n);

    /* Cleanup */
    free(in);
    free(out);

    return EXIT_SUCCESS;
}
