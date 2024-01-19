#include <stdio.h>
#include <stdlib.h>
#include <math.h> /* for fminf() */
#include <assert.h>
#include <mpi.h>

/* Compute the bounding box of |n| rectangles whose opposite vertices
   have coordinates (|x1[i]|, |y1[i]|), (|x2[i]|, |y2[i]|). The
   opposite corners of the bounding box will be stored in (|xb1|,
   |yb1|), (|xb2|, |yb2|) */
void bbox( const float *x1, const float *y1, const float* x2, const float *y2,
           int n,
           float *xb1, float *yb1, float *xb2, float *yb2 )
{
    int i;
    assert(n > 0);
    *xb1 = x1[0];
    *yb1 = y1[0];
    *xb2 = x2[0];
    *yb2 = y2[0];
    for (i=1; i<n; i++) {
        *xb1 = fminf( *xb1, x1[i] );
        *yb1 = fmaxf( *yb1, y1[i] );
        *xb2 = fmaxf( *xb2, x2[i] );
        *yb2 = fminf( *yb2, y2[i] );
    }
}

int main( int argc, char* argv[] )
{
    float *x1, *y1, *x2, *y2;
    float xb1, yb1, xb2, yb2, realx1, realx2, realy1, realy2;
    int N;
    int my_rank, comm_sz;
    //float *my_x1, *my_y1, *my_x2, *my_y2;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);

    if ( (0 == my_rank) && (argc != 2) ) {
        printf("Usage: %s [inputfile]\n", argv[0]);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    x1 = y1 = x2 = y2 = NULL;

    if ( 0 == my_rank ) {
        FILE *in = fopen(argv[1], "r");
        int i;
        if ( in == NULL ) {
            fprintf(stderr, "Cannot open %s for reading\n", argv[1]);
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }
        if (1 != fscanf(in, "%d", &N)) {
            fprintf(stderr, "FATAL: cannot read number of boxes\n");
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }

        /* Remove the following check if your implementation supports
           every value of N */
        if (N % comm_sz) {
            fprintf(stderr, "The number of rectangles (%d) must be a multiple of the communicator size (%d)\n", N, comm_sz);
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }

        x1 = (float*)malloc(N * sizeof(*x1)); assert(x1 != NULL);
        y1 = (float*)malloc(N * sizeof(*y1)); assert(y1 != NULL);
        x2 = (float*)malloc(N * sizeof(*x2)); assert(x2 != NULL);
        y2 = (float*)malloc(N * sizeof(*y2)); assert(y2 != NULL);

        for (i=0; i<N; i++) {
            if (4 != fscanf(in, "%f %f %f %f", &x1[i], &y1[i], &x2[i], &y2[i])) {
                fprintf(stderr, "FATAL: cannot read box %d\n", i);
                MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
            }
            assert(x1[i] < x2[i]);
            assert(y1[i] > y2[i]);
        }
        fclose(in);
    }

    MPI_Bcast( &N, 1, MPI_INT, 0, MPI_COMM_WORLD );

    if ( my_rank != 0 ) {
        x1 = (float*)malloc(N * sizeof(*x1)); assert(x1 != NULL);
        y1 = (float*)malloc(N * sizeof(*y1)); assert(y1 != NULL);
        x2 = (float*)malloc(N * sizeof(*x2)); assert(x2 != NULL);
        y2 = (float*)malloc(N * sizeof(*y2)); assert(y2 != NULL);
    }

    MPI_Scatter( x1, N / comm_sz, MPI_FLOAT, x1, N / comm_sz, MPI_FLOAT, 0, MPI_COMM_WORLD );
    MPI_Scatter( x2, N / comm_sz, MPI_FLOAT, x2, N / comm_sz, MPI_FLOAT, 0, MPI_COMM_WORLD );
    MPI_Scatter( y1, N / comm_sz, MPI_FLOAT, y1, N / comm_sz, MPI_FLOAT, 0, MPI_COMM_WORLD );
    MPI_Scatter( y2, N / comm_sz, MPI_FLOAT, y2, N / comm_sz, MPI_FLOAT, 0, MPI_COMM_WORLD );

    /* Compute the bounding box */
    bbox( x1, y1, x2, y2, N, &xb1, &yb1, &xb2, &yb2 );

    MPI_Reduce( &xb1, &realx1, 1, MPI_FLOAT, MPI_MIN, 0, MPI_COMM_WORLD );
    MPI_Reduce( &xb2, &realx2, 1, MPI_FLOAT, MPI_MAX, 0, MPI_COMM_WORLD );
    MPI_Reduce( &yb1, &realy1, 1, MPI_FLOAT, MPI_MAX, 0, MPI_COMM_WORLD );
    MPI_Reduce( &yb2, &realy2, 1, MPI_FLOAT, MPI_MIN, 0, MPI_COMM_WORLD );

    /* Print bounding box */
    if ( my_rank == 0 ) {
        printf("bbox: %f %f %f %f\n", xb1, yb1, xb2, yb2);
    }
    
    /* Free the memory */
    free(x1);
    free(y1);
    free(x2);
    free(y2);
    MPI_Finalize();

    return EXIT_SUCCESS;
}
