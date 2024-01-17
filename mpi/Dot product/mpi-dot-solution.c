#include <stdio.h>
#include <stdlib.h>
#include <math.h> /* for fabs() */
#include <assert.h>
#include <mpi.h>

/*
 * Compute sum { x[i] * y[i] }, i=0, ... n-1
 */
double dot( const double* x, const double* y, int n )
{
    double s = 0.0;
    int i;
    for (i=0; i<n; i++) {
        s += x[i] * y[i];
    }
    return s;
}

int main( int argc, char* argv[] )
{
    const double TOL = 1e-5;
    double *x = NULL, *y = NULL, result = 0.0, my_result = 0.0;
    int i, n = 1000;
    int my_rank, comm_sz;
    double *my_x = NULL, *my_y = NULL;
    int *sendcounts, *displs;
    int rough_partition_length;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);

    if ( argc > 1 ) {
        n = atoi(argv[1]);
    }

    rough_partition_length = (n / comm_sz);

    sendcounts = (int*)malloc(sizeof(int)*comm_sz);
    assert(sendcounts != NULL);

    displs = (int*)malloc(sizeof(int)*comm_sz);
    assert(displs != NULL);

    my_x = (double*)calloc(rough_partition_length+(n % comm_sz), sizeof(double));
    assert(my_x != NULL);

    my_y = (double*)calloc(rough_partition_length+(n % comm_sz), sizeof(double));
    assert(my_y != NULL);

    if ( 0 == my_rank ) {
        /* The master allocates the vectors */
        x = (double*)malloc( n * sizeof(*x) ); assert(x != NULL);
        y = (double*)malloc( n * sizeof(*y) ); assert(y != NULL);
        for ( i=0; i<n; i++ ) {
            x[i] = i + 1.0;
            y[i] = 1.0 / x[i];
        }

        int start = 0;
        int partition_size;
        for ( i=0; i<comm_sz; i++ ) {
            partition_size = i < (n % comm_sz) ? rough_partition_length : rough_partition_length + 1;
            sendcounts[i] = partition_size;
            displs[i] = start;
            start+=partition_size;
        }       

    }

    MPI_Scatterv(x, sendcounts, displs, MPI_DOUBLE, my_x, rough_partition_length+(n % comm_sz), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatterv(y, sendcounts, displs, MPI_DOUBLE, my_y, rough_partition_length+(n % comm_sz), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    my_result = dot(my_x, my_y, rough_partition_length+(n % comm_sz));
    
    MPI_Reduce(&my_result, &result, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if (0 == my_rank) {
        printf("Dot product: %f\n", result);
        if ( fabs(result - n) < TOL ) {
            printf("Check OK\n");
        } else {
            printf("Check failed: got %f, expected %f\n", result, (double)n);
        }
    }

    free(x); /* if x == NULL, does nothing */
    free(y);
    free(my_x);
    free(my_y);
    free(sendcounts);
    free(displs);

    MPI_Finalize();

    return EXIT_SUCCESS;
}
