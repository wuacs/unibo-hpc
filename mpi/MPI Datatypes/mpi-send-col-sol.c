#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

/* Matrix size */
#define SIZE 4

/* Initialize matrix m with the values k, k+1, k+2, ..., from left to
   right, top to bottom. m must point to an already allocated block of
   (size+2)*size integers. The first and last column of m is the halo,
   which is set to -1. */
void init_matrix( int *m, int size, int k )
{
    int i, j;
    for (i=0; i<size; i++) {
        for (j=0; j<size+2; j++) {
            if ( j==0 || j==size+1) {
                m[i*(size+2)+j] = -1;
            } else {
                m[i*(size+2)+j] = k;
                k++;
            }
        }
    }
}

void print_matrix( int *m, int size )
{
    int i, j;
    for (i=0; i<size; i++) {
        for (j=0; j<size+2; j++) {
            printf("%3d ", m[i*(size+2)+j]);
        }
        printf("\n");
    }
}

int main( int argc, char *argv[] )
{
    int my_rank, comm_sz;
    int my_mat[SIZE][SIZE+2];
    int partner;

    MPI_Init( &argc, &argv );
    MPI_Comm_rank( MPI_COMM_WORLD, &my_rank );
    MPI_Comm_size( MPI_COMM_WORLD, &comm_sz );

    MPI_Datatype MPI_Column;
    MPI_Type_vector(SIZE, 1, SIZE+2, MPI_INT, &MPI_Column);
    MPI_Type_commit(&MPI_Column);

    if ( 0 == my_rank && 2 != comm_sz ) {
        fprintf(stderr, "You must execute exactly 2 processes\n");
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
    if ( 0 == my_rank ) {
        init_matrix(&my_mat[0][0], SIZE, 0);

    } else if (1 == my_rank) {
        init_matrix(&my_mat[0][0], SIZE, SIZE*SIZE);
    }

    partner = (my_rank + 1) % 2;

    MPI_Sendrecv(
                    &my_mat[0][SIZE],       /* Sendbuffer */
                    1,                      /* SendCount */
                    MPI_Column,             /* Sendtype */
                    partner,                /* dest */
                    0,                      /* sendtag */
                    &my_mat[0][0],          /* recvbuffer */
                    1,                      /* recvcount */
                    MPI_Column,             /* recvtype */
                    partner,                /* source */
                    0,                      /* recvtag */
                    MPI_COMM_WORLD,         /* Communicator */
                    MPI_STATUS_IGNORE       /* Status */
                );

    MPI_Sendrecv(
                    &my_mat[0][1],          /* Sendbuffer */
                    1,                /* SendCount */
                    MPI_Column,             /* Sendtype */
                    partner,                      /* dest */
                    0,                      /* sendtag */
                    &my_mat[0][SIZE+1],     /* recvbuffer */
                    1,                /* recvcount */
                    MPI_Column,             /* recvtype */
                    partner,                      /* source */
                    0,                      /* recvtag */
                    MPI_COMM_WORLD,         /* Communicator */
                    MPI_STATUS_IGNORE       /* Status */
                );

    /* Print the matrices after the exchange; to do so without
       interference we must use this funny strategy: process 0 prints,
       then the processes synchronize, then process 1 prints. */
    if ( 0 == my_rank ) {
        printf("\n\nProcess 0:\n");
        print_matrix(&my_mat[0][0], SIZE);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    if ( 1 == my_rank ) {
        printf("\n\nProcess 1:\n");
        print_matrix(&my_mat[0][0], SIZE);
    }

    MPI_Finalize();
    return EXIT_SUCCESS;
}