#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <assert.h>
#include <mpi.h>

const int MAXIT = 100;

/* The __attribute__(( ... )) definition is gcc-specific, and tells
   the compiler that the fields of this structure should not be padded
   or aligned in any way. */
typedef struct __attribute__((__packed__)) {
    uint8_t r;  /* red   */
    uint8_t g;  /* green */
    uint8_t b;  /* blue  */
} pixel_t;

/* color gradient from https://stackoverflow.com/questions/16500656/which-color-gradient-is-used-to-color-mandelbrot-in-wikipedia */
const pixel_t colors[] = {
    {66, 30, 15}, /* r, g, b */
    {25, 7, 26},
    {9, 1, 47},
    {4, 4, 73},
    {0, 7, 100},
    {12, 44, 138},
    {24, 82, 177},
    {57, 125, 209},
    {134, 181, 229},
    {211, 236, 248},
    {241, 233, 191},
    {248, 201, 95},
    {255, 170, 0},
    {204, 128, 0},
    {153, 87, 0},
    {106, 52, 3} };
const int NCOLORS = sizeof(colors)/sizeof(colors[0]);

/*
 * Iterate the recurrence:
 *
 * z_0 = 0;
 * z_{n+1} = z_n^2 + cx + i*cy;
 *
 * Returns the first `n` such that `z_n > bound`, or `MAXIT` if `z_n` is below
 * `bound` after `MAXIT` iterations.
 */
int iterate( float cx, float cy )
{
    float x = 0.0f, y = 0.0f, xnew, ynew;
    int it;
    for ( it = 0; (it < MAXIT) && (x*x + y*y <= 2.0*2.0); it++ ) {
        xnew = x*x - y*y + cx;
        ynew = 2.0*x*y + cy;
        x = xnew;
        y = ynew;
    }
    return it;
}

/* Draw the rows of the Mandelbrot set from `ystart` (inclusive) to
   `yend` (excluded) to the bitmap pointed to by `p`. Note that `p`
   must point to the beginning of the bitmap where the portion of
   image will be stored; in other words, this function writes to
   pixels p[0], p[1], ... `xsize` and `ysize` MUST be the sizes
   of the WHOLE image. */
void draw_lines( int ystart, int yend, pixel_t* p, int xsize, int ysize )
{
    int x, y;
    for ( y = ystart; y < yend; y++) {
        for ( x = 0; x < xsize; x++ ) {
            const float cx = -2.5 + 3.5 * (float)x / (xsize - 1);
            const float cy = 1 - 2.0 * (float)y / (ysize - 1);
            const int v = iterate(cx, cy);
            if (v < MAXIT) {
                p->r = colors[v % NCOLORS].r;
                p->g = colors[v % NCOLORS].g;
                p->b = colors[v % NCOLORS].b;
            } else {
                p->r = p->g = p->b = 0;
            }
            p++;
        }
    }
}

int main( int argc, char *argv[] )
{
    int my_rank, comm_sz;
    FILE *out = NULL;
    const char* fname="mpi-mandelbrot.ppm";
    pixel_t *bitmap = NULL;
    pixel_t *partition_bitmap = NULL;
    int xsize, ysize;
    int my_y_start, my_y_end;
    int portion;
    int *recvcounts, *displs;
    int my_recvcount, my_displ;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);

    if ( argc > 1 ) {
        ysize = atoi(argv[1]);
    } else {
        ysize = 1024;
    }

    xsize = ysize * 1.4;
    

    portion = (xsize * ysize / comm_sz) + xsize; /* How many pixels each proc is going to process*/
    
    partition_bitmap = (pixel_t*)malloc(portion*sizeof(*bitmap));
    assert(partition_bitmap != NULL);


    if ( 0 == my_rank ) {
        /* Allocate the complete bitmap */
        bitmap = (pixel_t*)malloc(xsize*ysize*sizeof(*bitmap));
        assert(bitmap != NULL);

        recvcounts = (int*)malloc(comm_sz*sizeof(int));
        assert(recvcounts != NULL);

        displs = (int*)malloc(comm_sz*sizeof(int));
        assert(displs != NULL);

        int shifts = (ysize%comm_sz)?1:0;
        displs[0] = 0;
        recvcounts[0] = (ysize/comm_sz + shifts)*xsize*3;

        for (int i=1; i<comm_sz; i++) {
            int y_index_start = (ysize/comm_sz)*i + shifts;
            int number_of_y_processed = (ysize/comm_sz);

            if (i < (ysize%comm_sz)) {
                number_of_y_processed++;
                shifts++;
            }
            displs[i]=y_index_start*xsize*3;
            recvcounts[i]=number_of_y_processed*xsize*3;
        }

    }

    MPI_Scatter(displs, 1, MPI_INT, &my_displ, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Scatter(recvcounts, 1, MPI_INT, &my_recvcount, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);


    my_y_start = my_displ/(3*xsize);
    my_y_end = my_y_start + my_recvcount/(3*xsize);

    /* Does the actual computation*/
    draw_lines(my_y_start, my_y_end, partition_bitmap, xsize, ysize);
    /*
    int MPI_Gatherv(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
                void *recvbuf, const int *recvcounts, const int *displs,
                MPI_Datatype recvtype, int root, MPI_Comm comm)
    */
    MPI_Gatherv(partition_bitmap, my_recvcount, MPI_BYTE, bitmap, recvcounts, displs, MPI_BYTE, 0, MPI_COMM_WORLD);


    if ( 0 == my_rank ) {
        out = fopen(fname, "w");
        if ( !out ) {
            fprintf(stderr, "Error: cannot create %s\n", fname);
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }

        /* Write the header of the output file */
        fprintf(out, "P6\n");
        fprintf(out, "%d %d\n", xsize, ysize);
        fprintf(out, "255\n");

        fwrite(bitmap, sizeof(*bitmap), xsize*ysize, out);
        fclose(out);
        free(bitmap);
    }

    free(partition_bitmap);

    MPI_Finalize();

    return EXIT_SUCCESS;
}
