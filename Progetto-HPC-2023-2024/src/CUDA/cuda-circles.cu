#include "hpc.h"
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>

#define MAXSIZEBLOCK 1024 

typedef struct {
    float x, y;   /* coordinates of center */
    float r;      /* radius */
    float dx, dy; /* displacements due to interactions with other circles */
} circle_t;

/* These constants can be replaced with #define's if necessary */
const float XMIN = 0.0;
const float XMAX = 1000.0;
const float YMIN = 0.0;
const float YMAX = 1000.0;
const float RMIN = 10.0;
const float RMAX = 100.0;
const float EPSILON = 1e-5;
const float K = 1.5;

/**
 * Return a random float in [a, b]
 */
float randab(float a, float b)
{
    return a + (((float)rand())/RAND_MAX) * (b-a);
}

/**
 * Creates and populates a circles array passed as the second parameter.
 */
void init_circles(int n, circle_t * circles)
{
    assert(circles != NULL);
    for (int i=0; i<n; i++) {
        circles[i].x = randab(XMIN, XMAX);
        circles[i].y = randab(YMIN, YMAX);
        circles[i].r = randab(RMIN, RMAX);
        circles[i].dx = circles[i].dy = 0.0;
    }
}

/**
 * Set all displacements to zero.
 */
__global__ void reset_displacements( circle_t * circles, int n )
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        circles[idx].dx = circles[idx].dy = 0.0;
    }
}

/**
* Reset the count of intersections counted by each circle.
*/
__global__ void reset_intersections( int * intersections, int n ) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        intersections[idx] = 0;
    }
}

__global__ void ker_compute_forces(circle_t * circles, int n, int * intersections) {

    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx < n) {
        return;
    }

    for (int j=0; j<n; j++) {
        if (idx == j) {
            continue;
        }
        const float deltax = circles[j].x - circles[idx].x;
        const float deltay = circles[j].y - circles[idx].y;
        const float dist = hypotf(deltax, deltay);
        const float Rsum = circles[idx].r + circles[j].r;
        if (dist < Rsum - EPSILON) {
            const float overlap = Rsum - dist;
            assert(overlap > 0.0);
            // avoid division by zero
            const float overlap_x = overlap / (dist + EPSILON) * deltax;
            const float overlap_y = overlap / (dist + EPSILON) * deltay;
            if (idx<j) {
                intersections[idx]++;
            }
            circles[idx].dx += -(overlap_x / K);
            circles[idx].dy += -(overlap_y / K);
        }
    }
    circles[idx].dx = block_circles[idx].dx;
    circles[idx].dy = block_circles[idx].dy;
    
}

/**
 * Move the circles to a new position according to the forces acting
 * on each one.
 */
__global__ void move_circles( circle_t * circles , int n)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        circles[idx].x += circles[idx].dx;
        circles[idx].y += circles[idx].dy;
    }
}

#ifdef MOVIE
/**
 * Dumps the circles into a text file that can be processed using
 * gnuplot. This function may be used for debugging purposes, or to
 * produce a movie of how the algorithm works.
 *
 * You may want to completely remove this function from the final
 * version.
 */
void dump_circles( circle_t * circles, int n, int iterno )
{
    char fname[64];
    snprintf(fname, sizeof(fname), "movie/circles-%05d.gp", iterno);
    FILE *out = fopen(fname, "w");
    const float WIDTH = XMAX - XMIN;
    const float HEIGHT = YMAX - YMIN;
    fprintf(out, "set term png notransparent large\n");
    fprintf(out, "set output \"circles-%05d.png\"\n", iterno);
    fprintf(out, "set xrange [%f:%f]\n", XMIN - WIDTH*.2, XMAX + WIDTH*.2 );
    fprintf(out, "set yrange [%f:%f]\n", YMIN - HEIGHT*.2, YMAX + HEIGHT*.2 );
    fprintf(out, "set size square\n");
    fprintf(out, "plot '-' with circles notitle\n");
    for (int i=0; i<n; i++) {
        fprintf(out, "%f %f %f\n", circles[i].x, circles[i].y, circles[i].r);
    }
    fprintf(out, "e\n");
    fclose(out);
}
#endif

int main( int argc, char* argv[] )
{
    int n = 10000;
    int iterations = 20;
    dim3 grid, block;
    circle_t * d_circles;
    int * d_intersections, *intersections; /* Used by CUDA Threads to count number of intersections without race condition. */
    size_t size; /* Size in bytes of the circles */
    size_t size_number; /* Size in bytes of number of circles * number of bytes for a integer. */
    circle_t * circles; /* Used only by dump circles AND for initialization. */

    if ( argc > 3 ) {
        fprintf(stderr, "Usage: %s [ncircles [iterations]]\n", argv[0]);
        return EXIT_FAILURE;
    }

    if (argc > 1) {
        n = atoi(argv[1]);
    }

    if (argc > 2) {
        iterations = atoi(argv[2]);
    }

    srand(3);

    grid.x = (n + MAXSIZEBLOCK - 1)/MAXSIZEBLOCK;
    block.x = MAXSIZEBLOCK;
    size = sizeof(circle_t)*n;
    size_number = sizeof(int)*n;

    intersections = (int*)malloc(size_number);
    assert(intersections!=NULL);
    circles = (circle_t*)malloc(size); /* Used only by dump circles AND for initialization. */
    assert(circles!=NULL);

    cudaSafeCall( cudaMalloc((void**)&d_circles, size) );
    cudaSafeCall( cudaMalloc((void**)&d_intersections, size_number) );

    init_circles(n, circles);
    
    cudaSafeCall( cudaMemcpy(d_circles, circles, size, cudaMemcpyHostToDevice) );

#ifndef MOVIE
    free(circles); /* Istantly frees circles if it isn't used anymore. */
#endif 
    const double tstart_prog = hpc_gettime();
#ifdef MOVIE
    dump_circles(circles, n, 0);
#endif
    for (int it=0; it<iterations; it++) {
        const double tstart_iter = hpc_gettime();
        reset_intersections<<<grid, block>>>(d_intersections, n);
        reset_displacements<<<grid, block>>>(d_circles, n);
        ker_compute_forces<<<grid, block>>>(d_circles, n, d_intersections);
        cudaSafeCall( cudaMemcpy(intersections, d_intersections, size_number, cudaMemcpyDeviceToHost) );
        int n_overlaps = 0;
        for (int k = 0; k<n; k++) {
            n_overlaps += intersections[k];
        } 
        move_circles<<<grid, block>>>(d_circles, n);
        const double elapsed_iter = hpc_gettime() - tstart_iter;
#ifdef MOVIE
        cudaSafeCall( cudaMemcpy(circles, d_circles, size, cudaMemcpyDeviceToHost) );
        dump_circles(circles, n, it+1);
#endif
        printf("Iteration %d of %d, %d overlaps (%f s)\n", it+1, iterations, n_overlaps, elapsed_iter);
    }
    const double elapsed_prog = hpc_gettime() - tstart_prog;
    printf("Elapsed time: %f\n", elapsed_prog);

    free(intersections);
#ifdef MOVIE
    free(circles);
#endif
    cudaSafeCall( cudaFree(d_intersections) );
    cudaSafeCall( cudaFree(d_circles) );
    return EXIT_SUCCESS;
}
