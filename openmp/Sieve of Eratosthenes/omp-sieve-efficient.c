#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

/* Mark all mutliples of `k` in the set {`from`, ..., `to`-1}; return
   how many numbers have been marked for the first time. `from` does
   not need to be a multiple of `k`, although in this program it
   always is. */
long mark( char *isprime, int k, long from, long to )
{
    long nmarked = 0l;
    
    from = ((from + k - 1)/k)*k; /* start from the lowest multiple of p that is >= from */
    #pragma omp parallel for reduction(+:nmarked)
    for ( long x=from; x<to; x+=k ) {
        if (isprime[x]) {
            isprime[x] = 0;
            nmarked++;
        }
    }
    return nmarked;
}

int main( int argc, char *argv[] )
{
    long n = 1000000l, nprimes, i;

    if ( argc > 2 ) {
        fprintf(stderr, "Usage: %s [n]\n", argv[0]);
        return EXIT_FAILURE;
    }

    if ( argc == 2 ) {
        n = atol(argv[1]);
    }

    if (n > (1ul << 31)) {
        fprintf(stderr, "FATAL: n too large\n");
        return EXIT_FAILURE;
    }

    char *isprime = (char*)malloc(n+1); assert(isprime != NULL);
    /* Initially, all numbers are considered primes */
    for (i=0; i<=n; i++)
        isprime[i] = 1;

    nprimes = n-1;
    const double tstart = omp_get_wtime();
    /* main iteration of the sieve */
    for (i=2; i*i <= n; i++) {
        if (isprime[i]) {
            nprimes -= mark(isprime, i, i*i, n+1);
        }
    }
    const double elapsed = omp_get_wtime() - tstart;
    /* Uncomment to print the list of primes */
    /*
    for (i=2; i<=n; i++) {
        if (isprime[i]) {printf("%ld ", i);}
    }
    printf("\n");
    */
    free(isprime);

    printf("There are %ld primes in {2, ..., %ld}\n", nprimes, n);

    printf("Elapsed time: %f\n", elapsed);

    return EXIT_SUCCESS;
}
