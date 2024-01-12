/* The rand_r() function is available only if _XOPEN_SOURCE=600 */
#define _XOPEN_SOURCE 600
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h> /* for fabs */
#include <assert.h>

/* Generate `n` random points within the square with corners (-1, -1),
   (1, 1); return the number of points that fall inside the circle
   centered ad the origin with radius 1.
   Uses `inside` to store in inside[i] the number of points that thread i
   generated which are considered inside the square.
    */
unsigned int generate_points( unsigned int n, unsigned long int* inside , int num_threads)
{
    unsigned long int n_inside = 0;
    unsigned int my_seed;
    unsigned long int my_start; /* inclusive */
    unsigned long int my_end; /* exclusive */
    unsigned long int my_id; /* thread id as returned from omp_get_thread_num() */

    /* The C function rand() is _NOT_ thread-safe, since it uses a
       global (shared) seed. Therefore, it can not be used inside an
       parallel region. We use rand_r() with an explicit per-thread
       seed. However, this means that in general the result computed
       by this program will depend on the number of threads used, and
       not only on the number of points that are generated. */
    #pragma omp parallel default(none) private(my_start, my_end, my_seed, my_id) shared(n, inside, num_threads)
{
    my_id = omp_get_thread_num();
    my_seed = 17 + 19*my_id;
    my_start = (n * my_id)/num_threads;
    my_end = (n * (my_id + 1))/num_threads;
    for (unsigned long int i=my_start; i<my_end; i++) {
        /* Generate two random values in the range [-1, 1] */
        const double x = (2.0 * rand_r(&my_seed)/(double)RAND_MAX) - 1.0;
        const double y = (2.0 * rand_r(&my_seed)/(double)RAND_MAX) - 1.0;
        if ( x*x + y*y <= 1.0 ) {
            inside[my_id]++;
        }
    }
}
    for (unsigned long int j=0; j<num_threads; j++) {
        n_inside += inside[j];
        printf("inside[j]: %ld\n", inside[j]);
    }
    return n_inside;
}

int main( int argc, char *argv[] )
{
    unsigned int n_points = 10000;
    unsigned int n_inside;
    const double PI_EXACT = 3.14159265358979323846;
    const char* str_num_threads = getenv("OMP_NUM_THREADS");
    int num_threads;
    unsigned long int* inside;

    assert(str_num_threads!=NULL);
    num_threads = atol(str_num_threads);
    inside = (unsigned long int*)calloc(num_threads, sizeof(unsigned long int));

    if ( argc > 2 ) {
        fprintf(stderr, "Usage: %s [n_points]\n", argv[0]);
        return EXIT_FAILURE;
    }

    if ( argc > 1 ) {
        n_points = atol(argv[1]);
    }

    printf("Generating %u points...\n", n_points);
    const double tstart = omp_get_wtime();
    n_inside = generate_points(n_points, inside, num_threads);
    const double elapsed = omp_get_wtime() - tstart;
    const double pi_approx = 4.0 * n_inside / (double)n_points;
    printf("PI approximation %f, exact %f, error %f%%\n", pi_approx, PI_EXACT, 100.0*fabs(pi_approx - PI_EXACT)/PI_EXACT);
    printf("Elapsed time: %f\n", elapsed);
    free(inside);
    return EXIT_SUCCESS;
}
