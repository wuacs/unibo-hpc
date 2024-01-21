#include "hpc.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>

#define MAXBLOCKLEN 1024
#define BLOCKSIDELEN 32

__global__ void kernel_matsum(float *p, float *q, float*r, int n) {
    int bx = blockIdx.x * blockDim.x;
    int by = blockIdx.y * blockDim.y;
    int original_x = bx + threadIdx.x;
    int original_y = by + threadIdx.y;
    if ( original_x < n && original_y < n) {
        int pos = original_x*n+original_y;
        r[pos] = p[pos] + q[pos];
    }
}

void matsum( float *p, float *q, float *r, int n )
{
    int gridside = (n + BLOCKSIDELEN - 1)/BLOCKSIDELEN;
    dim3 grid(gridside, gridside);
    dim3 block(BLOCKSIDELEN, BLOCKSIDELEN);
    float *d_p, *d_q, *d_r;
    const size_t size = n*n*sizeof(float);

    cudaSafeCall( cudaMalloc((void**)&d_p, size) );
    cudaSafeCall( cudaMalloc((void**)&d_q, size) );
    cudaSafeCall( cudaMalloc((void**)&d_r, size) );

    cudaSafeCall(cudaMemcpy(d_p, p, size, cudaMemcpyHostToDevice) );
    cudaSafeCall(cudaMemcpy(d_q, q, size, cudaMemcpyHostToDevice) );
    cudaSafeCall(cudaMemcpy(d_r, r, size, cudaMemcpyHostToDevice) );

    kernel_matsum<<<grid, block>>>(d_p, d_q, d_r, n);
    cudaCheckError();

    cudaSafeCall(cudaMemcpy(r, d_r, size, cudaMemcpyDeviceToHost) );

    cudaFree(d_p);
    cudaFree(d_q);
    cudaFree(d_r);
}

/* Initialize square matrix p of size nxn */
void fill( float *p, int n )
{
    int i, j, k=0;
    for (i=0; i<n; i++) {
        for (j=0; j<n; j++) {
            p[i*n+j] = k;
            k = (k+1) % 1000;
        }
    }
}

/* Check result */
int check( float *r, int n )
{
    int i, j, k = 0;
    for (i=0; i<n; i++) {
        for (j=0; j<n; j++) {
            if (fabsf(r[i*n+j] - 2.0*k) > 1e-5) {
                fprintf(stderr, "Check FAILED: r[%d][%d] = %f, expeted %f\n", i, j, r[i*n+j], 2.0*k);
                return 0;
            }
            k = (k+1) % 1000;
        }
    }
    printf("Check OK\n");
    return 1;
}

int main( int argc, char *argv[] )
{
    float *p, *q, *r;
    int n = 1024;
    const int max_n = 5000;

    if ( argc > 2 ) {
        fprintf(stderr, "Usage: %s [n]\n", argv[0]);
        return EXIT_FAILURE;
    }

    if ( argc > 1 ) {
        n = atoi(argv[1]);
    }

    if ( n > max_n ) {
        fprintf(stderr, "FATAL: the maximum allowed matrix size is %d\n", max_n);
        return EXIT_FAILURE;
    }

    printf("Matrix size: %d x %d\n", n, n);

    const size_t size = n*n*sizeof(*p);

    /* Allocate space for p, q, r */
    p = (float*)malloc(size); assert(p != NULL);
    fill(p, n);
    q = (float*)malloc(size); assert(q != NULL);
    fill(q, n);
    r = (float*)malloc(size); assert(r != NULL);

    const double tstart = hpc_gettime();
    matsum(p, q, r, n);
    const double elapsed = hpc_gettime() - tstart;

    printf("Elapsed time (including data movement): %f\n", elapsed);
    printf("Throughput (Melements/s): %f\n", n*n/(1e6 * elapsed));

    /* Check result */
    check(r, n);

    /* Cleanup */
    free(p);
    free(q);
    free(r);

    return EXIT_SUCCESS;
}
