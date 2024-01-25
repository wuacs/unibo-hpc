#include "hpc.h"
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>

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

int ncircles;
circle_t *circles = NULL;

/**
 * Return a random float in [a, b]
 */
float randab(float a, float b)
{
    return a + (((float)rand())/RAND_MAX) * (b-a);
}

/**
 * Create and populate the array `circles[]` with randomly placed
 * circls.
 *
 * Do NOT parallelize this function.
 */
void init_circles(int n)
{
    assert(circles == NULL);
    ncircles = n;
    circles = (circle_t*)malloc(n * sizeof(*circles));
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
void reset_displacements( void )
{
    for (int i=0; i<ncircles; i++) {
        circles[i].dx = circles[i].dy = 0.0;
    }
}

void iterate(const int start_index, float** dx, float** dy, int * n_intersections, int my_rank) {
    //iteration_done[my_rank]++;
    int i = start_index;
    for (int j=i+1; j<ncircles; j++) {
        //iteration_done[my_rank]++;
        const float deltax = circles[j].x - circles[i].x;
        const float deltay = circles[j].y - circles[i].y;
        const float dist = hypotf(deltax, deltay);
        const float Rsum = circles[i].r + circles[j].r;
        if (dist < Rsum - EPSILON) {
            n_intersections[my_rank]++;
            const float overlap = Rsum - dist;
            assert(overlap > 0.0);
            // avoid division by zero
            const float overlap_x = overlap / (dist + EPSILON) * deltax;
            const float overlap_y = overlap / (dist + EPSILON) * deltay;
            dx[my_rank][i] += -(overlap_x / K);
            dy[my_rank][i] += -(overlap_y / K);
            dx[my_rank][j] += (overlap_x / K);
            dy[my_rank][j] += (overlap_y / K);
        }
    }
}

/**
 * Compute the force acting on each circle; returns the number of
 * overlapping pairs of circles (each overlapping pair must be counted
 * only once).
 */
int compute_forces( int thread_num)
{
    int n_intersections = 0;

    const double tstart_iter_total = hpc_gettime();

    int my_rank, my_first, my_last;

    float **dx = (float**)malloc(thread_num*sizeof(*dx));
    float **dy = (float**)malloc(thread_num*sizeof(*dy));

    for (int i = 0; i < thread_num; i++) {
        dx[i] = (float*)calloc(ncircles, sizeof(**dx));
        dy[i] = (float*)calloc(ncircles, sizeof(**dy));
    }

    int *private_intersection = (int*)calloc(thread_num, sizeof(*private_intersection));
    int left_border, right_border;
    
    left_border = ncircles / 2 + 1;
    right_border = ncircles / 2;

    #pragma omp parallel shared(dx, dy, ncircles, private_intersection, circles, EPSILON, K, thread_num, left_border, right_border) private(my_first, my_last, my_rank) default(none)
{
    my_rank = omp_get_thread_num();
    my_first = my_rank;
    my_last = ncircles - 1 - my_first;
    while (my_first < left_border || (my_last > right_border)) {
        if ( my_first < left_border ) {
            iterate(my_first, dx, dy, private_intersection, my_rank);
        }
        if ( my_last > right_border) {
            iterate(my_last, dx, dy, private_intersection, my_rank);
        }
        my_first = my_first + thread_num;
        my_last = ncircles - 1 - my_first;
    }
}

    #pragma omp parallel for schedule(static, ncircles) default(none) shared(ncircles, thread_num, dx, dy, circles, private_intersection) reduction(+:n_intersections)
    for (long long i = 0; i < thread_num; i++) {
        for (long long j = 0; j < ncircles; j++) {
            circles[j].dx+=dx[i][j];
            circles[j].dy+=dy[i][j];
        }
        n_intersections+=private_intersection[i];
    }

    const double total_elapsed_iter = hpc_gettime() - tstart_iter_total;

    printf("total iteration took: (%f s)\n", total_elapsed_iter);

    for (int i = 0; i < thread_num; i++) {
        free(dx[i]);
        free(dy[i]);
    }
    
    free(dx);
    free(dy);
    free(private_intersection);

    return n_intersections;
}

/**
 * Move the circles to a new position according to the forces acting
 * on each one.
 */
void move_circles( void )
{
    //#pragma omp parallel for default(none) shared(circles, ncircles)
    for (int i=0; i<ncircles; i++) {
        circles[i].x += circles[i].dx;
        circles[i].y += circles[i].dy;
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
void dump_circles( int iterno )
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
    for (int i=0; i<ncircles; i++) {
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
    const char* str_num_threads = getenv("OMP_NUM_THREADS");
    int num_threads;

    assert(str_num_threads!=NULL);
    num_threads = atoi(str_num_threads);
    
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

    init_circles(n);
    const double tstart_prog = hpc_gettime();
#ifdef MOVIE
    dump_circles(0);
#endif
    for (int it=0; it<iterations; it++) {
        const double tstart_iter = hpc_gettime();
        reset_displacements();
        const int n_overlaps = compute_forces(num_threads);
        move_circles();
        const double elapsed_iter = hpc_gettime() - tstart_iter;
#ifdef MOVIE
        dump_circles(it+1);
#endif
        printf("Iteration %d of %d, %d overlaps (%f s)\n", it+1, iterations, n_overlaps, elapsed_iter);
    }
    const double elapsed_prog = hpc_gettime() - tstart_prog;
    printf("Elapsed time: %f\n", elapsed_prog);

    free(circles);

    return EXIT_SUCCESS;
}
