#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <omp.h>

/* Recursive computation of the n-th Fibonacci number, for n=0, 1, 2, ...
   Do not parallelize this function. */
int fib_rec(int n)
{
    if (n<2) {
        return 1;
    } else {
        return fib_rec(n-1) + fib_rec(n-2);
    }
}

/* Iterative computation of the n-th Fibonacci number. This function
   must be used for checking the result only. */
int fib_iter(int n)
{
    if (n<2) {
        return 1;
    } else {
        int fibnm1 = 1;
        int fibnm2 = 1;
        int fibn;
        n = n-1;
        do {
            fibn = fibnm1 + fibnm2;
            fibnm2 = fibnm1;
            fibnm1 = fibn;
            n--;
        } while (n>0);
        return fibn;
    }
}

/* Fill vectors `vin` and `vout` of length `n`; `vin` will contain
   input values; `vout` is initialized with -1 */
void fill(int *vin, int *vout, int n)
{
    int i;
    /* fill input array */
    for (i=0; i<n; i++) {
        vin[i] = 25 + (i%10);
        vout[i] = -1;
    }
}

/* Check correctness of `vout[]`. Return 1 if correct, 0 if not */
int is_correct(const int *vin, const int *vout, int n)
{
    int i;
    /* check result */
    for (i=0; i<n; i++) {
        if ( vout[i] != fib_iter(vin[i]) ) {
            fprintf(stderr,
                    "Test FAILED: vin[%d]=%d, vout[%d]=%d (expected %d)\n",
                    i, vin[i], i, vout[i], fib_iter(vin[i]));
            return 0;
        }
    }
    fprintf(stderr, "Test OK\n");
    return 1;
}


void do_static(const int *vin, int *vout, int n, int chunk_size)
{
    int i, j;
    int my_id;
    int num_threads;

    #pragma omp parallel default(none) private(my_id, i, j, num_threads) shared(n, vin, vout, chunk_size)
{
    my_id = omp_get_thread_num();
    num_threads = omp_get_num_threads();
    for (i=my_id; i<n; i+=num_threads) {
        for (j=i; j<i+chunk_size && j<n; j++) {
            vout[j] = fib_rec(vin[j]);
            /* printf("vin[%d]=%d vout[%d]=%d\n", j, vin[j], j, vout[j]); */
        }
    }
}
}

void do_dynamic(const int *vin, int *vout, int n, int chunk_size)
{
    int i;
    int idx=0;
    int my_idx;

    #pragma omp parallel default(none) private(i, my_idx) shared(n, vout, vin, idx, chunk_size)
{
    #pragma omp critical 
    {
        my_idx = idx;
        idx+=chunk_size;
    } 
    while (my_idx < n) {
        i = my_idx;
        vout[i] = fib_rec(vin[i]);
        /* printf("vin[%d]=%d vout[%d]=%d\n", i, vin[i], i, vout[i]); */
        #pragma omp critical 
        {
            my_idx = idx;
            idx+=chunk_size;
        }
    }
    
}
}

int main( int argc, char* argv[] )
{
    int n = 1000;
    int chunk_size = 1;
    const int max_n = 512*1024*1024;
    int *vin, *vout;
    double tstart, elapsed;

    if ( argc > 3 ) {
        fprintf(stderr, "Usage: %s [n] [chunk_size]\n", argv[0]);
        return EXIT_FAILURE;
    }

    if ( argc > 2 ) {
        chunk_size = atoi(argv[2]);
    }

    if ( argc > 1 ) {
        n = atoi(argv[1]);
    }

    if ( n > max_n ) {
        fprintf(stderr, "FATAL: n too large (max value is %d)\n", max_n);
        return EXIT_FAILURE;
    }

    /* initialize the input and output arrays */
    vin = (int*)malloc(n * sizeof(vin[0])); assert(vin != NULL);
    vout = (int*)malloc(n * sizeof(vout[0])); assert(vout != NULL);

    /**
     ** Test static schedule implementation
     **/
    fill(vin, vout, n);
    tstart = omp_get_wtime();
    do_static(vin, vout, n, chunk_size);
    elapsed = omp_get_wtime() - tstart;
    is_correct(vin, vout, n);

    printf("Elapsed time (static schedule): %f\n", elapsed);

    /**
     ** Test dynamic schedule implementation**/
     
    fill(vin, vout, n);
    tstart = omp_get_wtime();
    do_dynamic(vin, vout, n, chunk_size);
    elapsed = omp_get_wtime() - tstart;
    is_correct(vin, vout, n);
    
    printf("Elapsed time (dynamic schedule): %f\n", elapsed);

    free(vin);
    free(vout);
    return EXIT_SUCCESS;
}
