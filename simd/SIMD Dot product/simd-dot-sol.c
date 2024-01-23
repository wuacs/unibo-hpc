#define _XOPEN_SOURCE 600

#include "hpc.h"
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <strings.h> /* for bzero() */
#include <math.h> /* for fabs() */

typedef float v4f __attribute__((vector_size(16)));
#define VLEN (sizeof(v4f)/sizeof(float))

/* Returns the dot product of arrays x[] and y[] of legnth n */
float serial_dot(const float *x, const float *y, int n)
{
    double r = 0.0; /* use double here to avoid some nasty rounding errors */
    int i;
    for (i=0; i<n; i++) {
        r += x[i] * y[i];
    }
    return r;
}

/* Same as above, but using the vector datatype of GCC */
float simd_dot(const float *x, const float *y, int n)
{
    v4f accumulator;
    v4f *x_simd = (v4f*)x;
    v4f *y_simd = (v4f*)y;
    accumulator = (*x_simd) * (*y_simd);
    int i; float r = 0.0;
    for (i=VLEN; i<=n-VLEN; i+=VLEN) {
        accumulator+= (*x_simd) * (*y_simd); 
        x_simd++;
        y_simd++;
    }
    for ( ; i<n; i++) {
        r+=(x[i]*y[i]);
    }
    for (i=0; i<VLEN; i++) {
        r+=accumulator[i];
    }
    return r;
}

/* Initialize vectors x and y */
void fill(float* x, float* y, int n)
{
    int i;
    const float xx[] = {-2.0f, 0.0f, 4.0f, 2.0f};
    const float yy[] = { 1.0f/2.0, 0.0f, 1.0/16.0, 1.0f/2.0f};
    const size_t N = sizeof(xx)/sizeof(xx[0]);

    for (i=0; i<n; i++) {
        x[i] = xx[i % N];
        y[i] = yy[i % N];
    }
}

int main(int argc, char* argv[])
{
    const float TOL = 1e-5;
    const int nruns = 10; /* number of replications */
    int r, n = 10*1024*1024;
    double serial_elapsed, simd_elapsed;
    double tstart, tend;
    float *x, *y, serial_result, simd_result;
    int ret;

    if ( argc > 2 ) {
        fprintf(stderr, "Usage: %s [n]\n", argv[0]);
        return EXIT_FAILURE;
    }

    if ( argc > 1 ) {
        n = atoi(argv[1]);
    }

    assert(n > 0);

    const size_t size = n * sizeof(*x);

    assert( size < 1024*1024*200UL );
    ret = posix_memalign((void**)&x, __BIGGEST_ALIGNMENT__, size);
    assert( 0 == ret );
    ret = posix_memalign((void**)&y, __BIGGEST_ALIGNMENT__, size);
    assert( 0 == ret );

    printf("Array length = %d\n", n);

    fill(x, y, n);
    /* Collect execution time of serial version */
    serial_elapsed = 0.0;
    for (r=0; r<nruns; r++) {
        tstart = hpc_gettime();
        serial_result = serial_dot(x, y, n);
        tend = hpc_gettime();
        serial_elapsed += tend - tstart;
    }
    serial_elapsed /= nruns;

    fill(x, y, n);
    /* Collect execution time of the parallel version */
    simd_elapsed = 0.0;
    for (r=0; r<nruns; r++) {
        tstart = hpc_gettime();
        simd_result = simd_dot(x, y, n);
        tend = hpc_gettime();
        simd_elapsed += tend - tstart;
    }
    simd_elapsed /= nruns;

    printf("Serial: result=%f, avg. time=%f (%d runs)\n", serial_result, serial_elapsed, nruns);
    printf("SIMD  : result=%f, avg. time=%f (%d runs)\n", simd_result, simd_elapsed, nruns);

    if ( fabs(serial_result - simd_result) > TOL ) {
        fprintf(stderr, "Check FAILED\n");
        return EXIT_FAILURE;
    }

    printf("Speedup (serial/SIMD) %f\n", serial_elapsed / simd_elapsed);

    free(x);
    free(y);
    return EXIT_SUCCESS;
}
