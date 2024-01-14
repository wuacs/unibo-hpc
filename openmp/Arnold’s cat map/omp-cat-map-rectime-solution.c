#include "hpc.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <omp.h>

/* Compute the Greatest Common Divisor (GCD) of integers a>0 and b>0 */
int gcd(int a, int b)
{
    assert(a>0);
    assert(b>0);

    while ( b != a ) {
        if (a>b) {
            a = a-b;
        } else {
            b = b-a;
        }
    }
    return a;
}

/* compute the Least Common Multiple (LCM) of integers a>0 and b>0 */
int lcm(int a, int b)
{
    assert(a>0);
    assert(b>0);
    return (a / gcd(a, b))*b;
}

/**
 * Compute the recurrence time of Arnold's cat map applied to an image
 * of size (n*n). For each point (x,y), compute the minimum recurrence
 * time k(x,y). The minimum recurrence time for the whole image is the
 * Least Common Multiple of all k(x,y).
 */
int cat_map_rectime( int n )
{
    int y, x;
    int rec_it[n][n]; /* In of rec_it[i][j] we will have minimum recurrence time for pixel with x=i and y=j */
    int recurrence_n; 

    #pragma omp parallel for collapse(2) default(none) shared(rec_it, n)
    for (y = 0; y < n; y++) {
        for (x = 0; x < n; x++) {
            int k = 0;
            /* Compute the k-th iterate of pixel (x, y) */
            int xstart = x, ystart = y;
            int xcur = x, ycur = y;
            do {
                const int xnext = (2*xcur+ycur) % n;
                const int ynext = (xcur + ycur) % n;
                xcur = xnext;
                ycur = ynext;
                k++;
            } while (xcur != xstart || ycur != ystart);
            rec_it[x][y] = k;
        }
    }

    recurrence_n = rec_it[0][0];
    for (x = 0; x < n; x++) {
        for (y = 1; y < n; y++) {
            recurrence_n = lcm(recurrence_n, rec_it[x][y]);
        }
    }

    return recurrence_n;
}

int main( int argc, char* argv[] )
{
    int n, k;

    if ( argc != 2 ) {
        fprintf(stderr, "Usage: %s image_size\n", argv[0]);
        return EXIT_FAILURE;
    }
    n = atoi(argv[1]);
    const double tstart = omp_get_wtime();
    k = cat_map_rectime(n);
    const double elapsed = omp_get_wtime() - tstart;
    printf("%d\n", k);

    printf("Elapsed time: %f\n", elapsed);

    return EXIT_SUCCESS;
}
