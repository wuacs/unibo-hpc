#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

/* Mark all mutliples of `k` in the set {`from`, ..., `to`-1}; return
   how many numbers have been marked for the first time. `from` does
   not need to be a multiple of `k`, although in this program it
   always is. */
long mark( char *isprime, int k, long from, long to , int num_threads)
{
    int my_id;
    int i;
    long * my_nmarked = (long*) calloc(num_threads, sizeof(long));
    long nmarked = 0l;
    long my_from;
    long my_to;
    long n;
    from = ((from + k - 1)/k)*k;
    n = to-from;
    #pragma omp parallel default(none) private(my_id, my_from, my_to) shared(isprime, my_nmarked, n, num_threads, k, from, to) 
{
    my_id = omp_get_thread_num();
    my_from = ((n * my_id) / num_threads) + from;
    if (my_from % k!=0) {
        my_from = ((my_from / k) + 1)*k;
    }
    my_to = ((n * (my_id + 1)) / num_threads) + from;
    for ( long x=my_from; x < my_to && x < to; x+=k ) {
        if (isprime[x]) {
            isprime[x] = 0;
            my_nmarked[my_id]++;
        }
    }
}
    for (i = 0; i<num_threads; i++) {
        nmarked+=my_nmarked[i];
    }

    free(my_nmarked);

    return nmarked;
}

int main( int argc, char *argv[] )
{
    long n = 1000000l, nprimes, i;
    int num_threads;
    const char* str_num_threads = getenv("OMP_NUM_THREADS");

    assert(str_num_threads!=NULL);
    num_threads = atoi(str_num_threads);

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
            nprimes -= mark(isprime, i, i*i, n+1, num_threads);
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
