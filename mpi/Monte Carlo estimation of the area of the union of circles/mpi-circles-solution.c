#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <mpi.h>

/* Computes the square of x */
float sq(float x)
{
    return x*x;
}

/* Generate `k` random points inside the square (0,0) --
  (100,100). Return the number of points that fall inside at least one
  of the `n` circles with center (x[i], y[i]) and radius r[i].  The
  result must be <= k. */
int inside( const float* x, const float* y, const float *r, int n, int k )
{
    int i, np, c=0;
    for (np=0; np<k; np++) {
        const float px = 100.0*rand()/(float)RAND_MAX;
        const float py = 100.0*rand()/(float)RAND_MAX;
        for (i=0; i<n; i++) {
            if ( sq(px-x[i]) + sq(py-y[i]) <= sq(r[i]) ) {
                c++;
                break;
            }
        }
    }
    return c;
}

int main( int argc, char* argv[] )
{
    float *x = NULL, *y = NULL, *r = NULL;
    int N, K, c = 0, my_c, my_number_to_generate;
    int my_rank, comm_sz;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);

    /* Initialize the Random Number Generator (RNG) */
    srand(my_rank * 7 + 11);

    if ( (0 == my_rank) && (argc != 3) ) {
        fprintf(stderr, "Usage: %s [npoints] [inputfile]\n", argv[0]);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    K = atoi(argv[1]);

    
    /* The input file is read by the master only */
    if ( 0 == my_rank ) {
        FILE *in = fopen(argv[2], "r");
        int i;
        if ( in == NULL ) {
            fprintf(stderr, "FATAL: Cannot open \"%s\" for reading\n", argv[1]);
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }
        if (1 != fscanf(in, "%d", &N)) {
            fprintf(stderr, "FATAL: Cannot read the number of circles\n");
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }

        x = (float*)malloc(N * sizeof(*x)); assert(x != NULL);
        y = (float*)malloc(N * sizeof(*y)); assert(y != NULL);
        r = (float*)malloc(N * sizeof(*r)); assert(r != NULL);

        for (i=0; i<N; i++) {
            if (3 != fscanf(in, "%f %f %f", &x[i], &y[i], &r[i])) {
                fprintf(stderr, "FATAL: Cannot read circle %d\n", i);
                MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
            }
        }
        fclose(in);
    }

    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);

    const double tstart = MPI_Wtime();
    
    if ( my_rank != 0 ) {
        x = (float*)malloc(N * sizeof(*x)); assert(x != NULL);
        y = (float*)malloc(N * sizeof(*y)); assert(y != NULL);
        r = (float*)malloc(N * sizeof(*r)); assert(r != NULL);
    } 
    
    MPI_Bcast(x, N, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(y, N, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(r, N, MPI_FLOAT, 0, MPI_COMM_WORLD);

    my_number_to_generate = K / comm_sz;
    
    if (my_rank < (K % comm_sz)) {
        my_number_to_generate++;
    }

    my_c = inside(x, y, r, N, my_number_to_generate);

    MPI_Reduce(&my_c, &c, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    /* the master prints the result */
    if ( 0 == my_rank ) {
        printf("%d points, %d inside, area=%f\n", K, c, 1.0e6*c/K);
        const double elapsed = MPI_Wtime() - tstart;
        printf("Execution time (s): %f\n", elapsed);
    }

    free(x);
    free(y);
    free(r);

    MPI_Finalize();

    return EXIT_SUCCESS;
}
