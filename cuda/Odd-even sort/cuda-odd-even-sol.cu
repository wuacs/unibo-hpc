#include "hpc.h"
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#define MAXBLOCKLEN 1024

/* if *a > *b, swap them. Otherwise do nothing */
__device__ void cmp_and_swap( int* a, int* b )
{
    if ( *a > *b ) {
        int tmp = *a;
        *a = *b;
        *b = tmp;
    }
}

__global__ void odd_even_phase(int * v, const int n, const int phase) { /* phase can either be 0 for even phase or 1 for odd. */
    int idx = (blockIdx.x * MAXBLOCKLEN + threadIdx.x)*2 + phase;
    //printf("my idx is %d\n", idx);
    if ( idx < n && (idx + 1 < n)) {
        cmp_and_swap(&v[idx], &v[idx+1]);
    }
}

/* Odd-even transposition sort */
void odd_even_sort( int* v, int n )
{
    int *d_v;
    const size_t size = sizeof(int)*n;

    cudaSafeCall( cudaMalloc((void**)&d_v, size) );

    cudaSafeCall( cudaMemcpy(d_v, v, size, cudaMemcpyHostToDevice) );

    for (int phase = 0; phase < n; phase++) {
        odd_even_phase<<<(n/2 + MAXBLOCKLEN - 1)/MAXBLOCKLEN, MAXBLOCKLEN>>>(d_v, n, phase % 2);
        cudaCheckError();
    }

    cudaSafeCall( cudaMemcpy(v, d_v, size, cudaMemcpyDeviceToHost) );

    cudaFree(d_v);
}


/**
 * Return a random integer in the range [a..b]
 */
int randab(int a, int b)
{
    return a + (rand() % (b-a+1));
}

/**
 * Fill vector x with a random permutation of the integers 0..n-1
 */
void fill( int *x, int n )
{
    int i;
    for (i=0; i<n; i++) {
        x[i] = i;
    }
    for(i=0; i<n-1; i++) {
        const int j = randab(i, n-1);
        const int tmp = x[i];
        x[i] = x[j];
        x[j] = tmp;
    }
}

/**
 * Check correctness of the result
 */
int check( const int *x, int n )
{
    int i;
    for (i=0; i<n; i++) {
        if (x[i] != i) {
            fprintf(stderr, "Check FAILED: x[%d]=%d, expected %d\n", i, x[i], i);
            return 0;
        }
    }
    printf("Check OK\n");
    return 1;
}

int main( int argc, char *argv[] )
{
    int *x;
    int n = 128*1024;
    const int MAX_N = 512*1024*1024;
    double tstart, elapsed;

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

    const size_t SIZE = n * sizeof(*x);

    /* Allocate space for x on host */
    x = (int*)malloc(SIZE); assert(x != NULL);
    fill(x, n);

    tstart = hpc_gettime();
    odd_even_sort(x, n);
    elapsed = hpc_gettime() - tstart;
    printf("Sorted %d elements in %f seconds\n", n, elapsed);

    /* Check result */
    check(x, n);


    /* Cleanup */
    free(x);

    return EXIT_SUCCESS;
}
