#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

void my_Bcast(int *v)
{
    int my_rank, comm_sz;
    int dest_1, dest_2;

    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
    
    dest_1 = (2*my_rank+1);
    dest_2 = (2*my_rank+2);

    if (my_rank == 0) {
        MPI_Send(v, 1, MPI_INT, dest_1, 0, MPI_COMM_WORLD);
        MPI_Send(v, 1, MPI_INT, dest_2, 0, MPI_COMM_WORLD);
    } else {
        MPI_Recv(v, 1, MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        if (dest_1 < comm_sz) {
            MPI_Send(v, 1, MPI_INT, dest_1, 0, MPI_COMM_WORLD);
        }
        if (dest_2 < comm_sz) {
            MPI_Send(v, 1, MPI_INT, dest_1, 0, MPI_COMM_WORLD);
        }
    }
}


int main( int argc, char *argv[] )
{
    const int SENDVAL = 999; /* valore che viene inviato agli altri processi */
    int my_rank;
    int v;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    /* process 0 sets `v` to the value to be broadcast, while all
       other processes set `v` to -1. */
    if ( 0 == my_rank ) {
        v = SENDVAL;
    } else {
        v = -1;
    }

    printf("BEFORE: value of `v` at rank %d = %d\n", my_rank, v);

    my_Bcast(&v);

    if ( v == 999 ) {
        printf("OK: ");
    } else {
        printf("ERROR: ");
    }
    printf("value of `v` at rank %d = %d\n", my_rank, v);

    MPI_Finalize();

    return EXIT_SUCCESS;
}
