#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

int main( int argc, char *argv[] )
{
    int my_rank, comm_sz, K = 10;
    int start_K = K;
    int sending_integer, receiving_integer;
    int next_rank, prev_rank;
    int first_rank = 0, last_rank;
    int is_over = 0;
    MPI_Status status;

    MPI_Init( &argc, &argv );
    MPI_Comm_rank( MPI_COMM_WORLD, &my_rank );
    MPI_Comm_size( MPI_COMM_WORLD, &comm_sz );

    if ( argc > 1 ) {
        K = atoi(argv[1]);
    }

    last_rank = comm_sz - 1;
    next_rank = last_rank == my_rank ? 0 : my_rank + 1;
    prev_rank = first_rank == my_rank ? last_rank : my_rank - 1;

    if (my_rank == 0) {
        sending_integer = 1;
        MPI_Send(&sending_integer, 1, MPI_INT, 1, 0, MPI_COMM_WORLD);
    }

    while (!is_over) {
        MPI_Recv(&receiving_integer, 1, MPI_INT, prev_rank, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
        if (my_rank == 0) {
            K--;
        }
        if (K == 0) {
            /* This next instruction will be done once, by the core wiht rank 0, for initiating propagation of termination. */
            MPI_Send(&sending_integer, 1, MPI_INT, next_rank, 1, MPI_COMM_WORLD);
            /* Now, core with rank 0, knows he can terminate. (1 of 2 ways of exiting the loop)*/
            is_over = 1;
            printf("At the end of %d-th iteration unit %d has received %d \n", start_K, my_rank, receiving_integer);
        } else {
            if (status.MPI_TAG == 0) { /* 0 here corresponds to CONTINUE THE TRANSITION; 1 means propagate termination; */
                sending_integer = receiving_integer+1;
                MPI_Send(&sending_integer, 1, MPI_INT, next_rank, 0, MPI_COMM_WORLD);
            } else if (status.MPI_TAG == 1) {
                if (my_rank != last_rank) { /* process 0 is the first which propagates termination signal
                                         so there is no point in sending it again to it. */
                    MPI_Send(NULL, 0, MPI_INT, next_rank, 1, MPI_COMM_WORLD);
                }
                is_over = 1; /* 2 of 2 ways of exiting this loop: used by all the cores except the core with rank 0. */
            }
        }
        
    }

    MPI_Finalize();
    return EXIT_SUCCESS;
}
