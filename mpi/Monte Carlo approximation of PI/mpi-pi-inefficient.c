#include <mpi.h>
#include <stdio.h>
#include <stdlib.h> /* for rand() */
#include <time.h>   /* for time() */
#include <math.h>   /* for fabs() */

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* Generate `n` random points within the square with corners (-1, -1),
   (1, 1); return the number of points that fall inside the circle
   centered ad the origin with radius 1 */
int generate_points( int n )
{
    int i, n_inside = 0;
    for (i=0; i<n; i++) {
        const double x = (rand()/(double)RAND_MAX * 2.0) - 1.0;
        const double y = (rand()/(double)RAND_MAX * 2.0) - 1.0;
        if ( x*x + y*y < 1.0 ) {
            n_inside++;
        }
    }
    return n_inside;
}

int main( int argc, char *argv[] )
{
    int my_rank, comm_sz;
    int inside = 0, npoints = 1000000;
    double pi_approx;
    int my_points;
    int inside_points_received;
    int received = 0;

    MPI_Init( &argc, &argv );
    MPI_Comm_rank( MPI_COMM_WORLD, &my_rank );
    MPI_Comm_size( MPI_COMM_WORLD, &comm_sz );

    if ( argc > 1 ) {
        npoints = atoi(argv[1]);
    }

    /* Each process initializes the pseudo-random number generator; if
       we don't do this (or something similar), each process would
       produce the exact same sequence of pseudo-random numbers! */
    srand(my_rank * 11 + 7);

    my_points = my_rank < (npoints / comm_sz) ? (npoints / comm_sz) + 1 : (npoints / comm_sz);
    inside = generate_points(my_points);

    if (my_rank == 0) {
        while (received < comm_sz - 1) {
            MPI_Recv(&inside_points_received, 1, MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            inside+=inside_points_received;
            received++;
        }
        pi_approx = 4.0 * inside / (double)npoints;
        printf("PI approximation is %f (true value=%f, rel error=%.3f%%)\n", pi_approx, M_PI, 100.0*fabs(pi_approx-M_PI)/M_PI);
    } else {
        MPI_Send(&inside, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
    }
    
    MPI_Finalize();
    return EXIT_SUCCESS;
}
