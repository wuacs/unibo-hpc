#include "hpc.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>

#define MAX_NUM_THREAD_PER_BLOCK 1024 /* You may want to modify this based on your hardware(run deviceQuery for more info) */

__global__ void thread_dot(float *x, float *y, float* tmp, int n) {

    tmp[threadIdx.x] = 0.0f;
    for ( int i = threadIdx.x; i < n; i+=MAX_NUM_THREAD_PER_BLOCK) {
        tmp[threadIdx.x] += x[i] * y[i];
    }
} 

float dot( const float *x, const float *y, int n )
{
    size_t size = n * sizeof(float);
    float result = 0.0;
    /* Define a `float` array tmp[] of size BLKDIM on host memory */
    float tmp[MAX_NUM_THREAD_PER_BLOCK];
    /* Define pointers to copies of x, y and tmp in device memory */
    float *d_x, *d_y, *d_tmp;
    /* Allocate space for device copies of x,y */
    cudaMalloc( (void **)&d_x, size ); /* In Bytes */
    cudaMalloc( (void **)&d_y, size );
    cudaMalloc( (void **)&d_tmp, MAX_NUM_THREAD_PER_BLOCK*sizeof(float) );
    /* Copy x, y from host to device */
    cudaMemcpy( d_x, x, size, cudaMemcpyHostToDevice );
    cudaMemcpy( d_y, y, size, cudaMemcpyHostToDevice );
    /* Launch a suitable kernel on the GPU */
    thread_dot<<<1, MAX_NUM_THREAD_PER_BLOCK>>>(d_x, d_y, d_tmp, n);
    /* Copy the result back to host memory */
    cudaMemcpy(&tmp[0], d_tmp, MAX_NUM_THREAD_PER_BLOCK*sizeof(float), cudaMemcpyDeviceToHost);
    /* Perform the final reduction on the CPU */
    for ( int i = 0; i < n && i < MAX_NUM_THREAD_PER_BLOCK; i++ ) {
        result += tmp[i];
    }
    
    /* Free device memory */
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_tmp);

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
