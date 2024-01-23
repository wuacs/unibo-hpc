/* The following #define is required by posix_memalign() */
#define _XOPEN_SOURCE 600

#include "hpc.h"
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>  /* for assert() */
#include <strings.h> /* for bzero() */

/* This program works on double-precision numbers; therefore, we
   define a v4d vector datatype that contains four doubles in a SIMD
   array of 32 bytes (VLEN==4). */
typedef double v4d __attribute__((vector_size(32)));
#define VLEN (sizeof(v4d)/sizeof(double))

/* Fills n x n square matrix m */
void fill( double* m, int n )
{
    int i, j;
    for (i=0; i<n; i++) {
        for (j=0; j<n; j++) {
            m[i*n + j] = (i%10+j) / 10.0;
        }
    }
}

/* compute r = p * q, where p, q, r are n x n matrices. */
void scalar_matmul( const double *p, const double* q, double *r, int n)
{
    int i, j, k;

    for (i=0; i<n; i++) {
        for (j=0; j<n; j++) {
            double s = 0.0;
            for (k=0; k<n; k++) {
                s += p[i*n + k] * q[k*n + j];
            }
            r[i*n + j] = s;
        }
    }
}

/* Cache-efficient computation of r = p * q, where p. q, r are n x n
   matrices. This function allocates (and then releases) an additional n x n
   temporary matrix. */
void scalar_matmul_tr( const double *p, const double* q, double *r, int n)
{
    int i, j, k;
    double *qT = (double*)malloc( n * n * sizeof(*qT) );

    assert(qT != NULL);

    /* transpose q, storing the result in qT */
    for (i=0; i<n; i++) {
        for (j=0; j<n; j++) {
            qT[j*n + i] = q[i*n + j];
        }
    }

    /* multiply p and qT row-wise */
    for (i=0; i<n; i++) {
        for (j=0; j<n; j++) {
            double s = 0.0;
            for (k=0; k<n; k++) {
                s += p[i*n + k] * qT[j*n + k];
            }
            r[i*n + j] = s;
        }
    }

    free(qT);
}

double simd_dot(const double *x, const double *y, int n)
{
    v4d accumulator;
    v4d *x_simd = (v4d*)x;
    v4d *y_simd = (v4d*)y;
    accumulator = (*x_simd) * (*y_simd);
    x_simd++;
    y_simd++;
    int i; double r = 0.0;
    for (i=VLEN; i<=n-VLEN; i+=VLEN) {
        accumulator+= (*x_simd) * (*y_simd); 
        x_simd++;
        y_simd++;
    }
    for (i=0; i<VLEN; i++) {
        r+=accumulator[i];
    }
    return r;
}

/* SIMD version of the cache-efficient matrix-matrix multiply above.
   This function requires that n is a multiple of the SIMD vector
   length VLEN */
void simd_matmul_tr( const double *p, const double* q, double *r, int n)
{
    int i, j;
    double *qT;
    posix_memalign((void**)&qT, __BIGGEST_ALIGNMENT__, n * n * sizeof(*qT));

    assert(qT != NULL);

    /* transpose q, storing the result in qT */
    for (i=0; i<n; i++) {
        for (j=0; j<n; j++) {
            qT[j*n + i] = q[i*n + j];
        }
    }

    /* multiply p and qT row-wise */
    for (i=0; i<n; i++) {
        for (j=0; j<n; j++) {
            r[i*n + j] = simd_dot(&p[i*n], &qT[j*n], n);
        }
    }
}

int main( int argc, char* argv[] )
{
    int n = 512;
    double *p, *q, *r;
    double tstart, elapsed, tserial;
    int ret;

    if ( argc > 2 ) {
        fprintf(stderr, "Usage: %s [n]\n", argv[0]);
        return EXIT_FAILURE;
    }

    if ( argc > 1 ) {
        n = atoi(argv[1]);
    }

    if ( 0 != n % VLEN ) {
        fprintf(stderr, "FATAL: the matrix size must be a multiple of %d\n", (int)VLEN);
        return EXIT_FAILURE;
    }

    const size_t size = n*n*sizeof(*p);

    ret = posix_memalign((void**)&p, __BIGGEST_ALIGNMENT__,  size);
    assert( 0 == ret );
    ret = posix_memalign((void**)&q, __BIGGEST_ALIGNMENT__,  size);
    assert( 0 == ret );
    ret = posix_memalign((void**)&r, __BIGGEST_ALIGNMENT__,  size);
    assert( 0 == ret );

    fill(p, n);
    fill(q, n);
    printf("\nMatrix size: %d x %d\n\n", n, n);

    tstart = hpc_gettime();
    scalar_matmul(p, q, r, n);
    tserial = elapsed = hpc_gettime() - tstart;
    printf("Scalar\t\tr[0][0] = %f, Exec time = %f\n", r[0], elapsed);

    bzero(r, size);

    tstart = hpc_gettime();
    scalar_matmul_tr(p, q, r, n);
    elapsed = hpc_gettime() - tstart;
    printf("Transposed\tr[0][0] = %f, Exec time = %f (speedup vs scalar %.2fx)\n", r[0], elapsed, tserial/elapsed );

    bzero(r, size);

    tstart = hpc_gettime();
    simd_matmul_tr(p, q, r, n);
    elapsed = hpc_gettime() - tstart;
    printf("SIMD transposed\tr[0][0] = %f, Exec time = %f (speedup vs scalar %.2fx)\n", r[0], elapsed, tserial/elapsed);

    free(p);
    free(q);
    free(r);
    return EXIT_SUCCESS;
}
