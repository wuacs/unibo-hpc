#include "hpc.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <omp.h>

typedef struct {
    int width;   /* Width of the image (in pixels) */
    int height;  /* Height of the image (in pixels) */
    int maxgrey; /* Don't care (used only by the PGM read/write routines) */
    unsigned char *bmap; /* buffer of width*height bytes; each element represents the gray level of a pixel (0-255) */
} PGM_image;

const unsigned char WHITE = 255;
const unsigned char BLACK = 0;

/**
 * Initialize a PGM_image object: allocate space for a bitmap of size
 * `width` x `height`, and set all pixels to color `col`
 */
void init_pgm( PGM_image *img, int width, int height, unsigned char col )
{
    int i, j;

    assert(img != NULL);

    img->width = width;
    img->height = height;
    img->maxgrey = 255;
    img->bmap = (unsigned char*)malloc(width*height);
    assert(img->bmap != NULL);
    for (i=0; i<height; i++) {
        for (j=0; j<width; j++) {
            img->bmap[i*width + j] = col;
        }
    }
}

/**
 * Read a PGM file from file `f`. Warning: this function is not
 * robust: it may fail on legal PGM images, and may crash on invalid
 * files since no proper error checking is done.
 */
void read_pgm( FILE *f, PGM_image* img )
{
    char buf[1024];
    const size_t BUFSIZE = sizeof(buf);
    char *s;
    int nread;

    assert(f != NULL);
    assert(img != NULL);

    /* Get the file type (must be "P5") */
    s = fgets(buf, BUFSIZE, f);
    if (0 != strcmp(s, "P5\n")) {
        fprintf(stderr, "Wrong file type %s\n", buf);
        exit(EXIT_FAILURE);
    }
    /* Get any comment and ignore it; does not work if there are
       leading spaces in the comment line */
    do {
        s = fgets(buf, BUFSIZE, f);
    } while (s[0] == '#');
    /* Get width, height */
    sscanf(s, "%d %d", &(img->width), &(img->height));
    /* get maxgrey; must be less than or equal to 255 */
    s = fgets(buf, BUFSIZE, f);
    sscanf(s, "%d", &(img->maxgrey));
    if ( img->maxgrey > 255 ) {
        fprintf(stderr, "FATAL: maxgray=%d > 255\n", img->maxgrey);
        exit(EXIT_FAILURE);
    }
#if _XOPEN_SOURCE < 600
    img->bmap = (unsigned char*)malloc((img->width)*(img->height)*sizeof(unsigned char));
#else
    /* The pointer img->bmap must be properly aligned to allow aligned
       SIMD load/stores to work. */
    int ret = posix_memalign((void**)&(img->bmap), __BIGGEST_ALIGNMENT__, (img->width)*(img->height));
    assert( 0 == ret );
#endif
    assert(img->bmap != NULL);
    /* Get the binary data from the file */
    nread = fread(img->bmap, 1, (img->width)*(img->height), f);
    if ( (img->width)*(img->height) != nread ) {
        fprintf(stderr, "FATAL: error reading input: expecting %d bytes, got %d\n", (img->width)*(img->height), nread);
        exit(EXIT_FAILURE);
    }
}

/**
 * Write the image `img` to file `f`; if not NULL, use the string
 * `comment` as metadata.
 */
void write_pgm( FILE *f, const PGM_image* img, const char *comment )
{
    assert(f != NULL);
    assert(img != NULL);

    fprintf(f, "P5\n");
    fprintf(f, "# %s\n", comment != NULL ? comment : "");
    fprintf(f, "%d %d\n", img->width, img->height);
    fprintf(f, "%d\n", img->maxgrey);
    fwrite(img->bmap, 1, (img->width)*(img->height), f);
}

/**
 * Free the bitmap associated with image `img`; note that the
 * structure pointed to by `img` is NOT deallocated; only `img->bmap`
 * is.
 */
void free_pgm( PGM_image *img )
{
    assert(img != NULL);
    free(img->bmap);
    img->bmap = NULL; /* not necessary */
    img->width = img->height = img->maxgrey = -1;
}

/**
 * Compute the `k`-th iterate of the cat map for image `img`. The
 * width and height of the image must be equal. This function must
 * replace the bitmap of `img` with the one resulting after ierating
 * `k` times the cat map. To do so, the function allocates a temporary
 * bitmap with the same size of the original one, so that it reads one
 * pixel from the "old" image and copies it to the "new" image. After
 * each iteration of the cat map, the role of the two bitmaps are
 * exchanged.
 */
void cat_map( PGM_image* img, int k )
{
    int i, x, y;
    const int N = img->width;
    unsigned char *cur = img->bmap;
    unsigned char *next = (unsigned char*)malloc( N*N*sizeof(unsigned char) );
    unsigned char *tmp;

    assert(next != NULL);
    assert(img->width == img->height);

    /* [TODO] Which of the following loop(s) can be parallelized? */
    for (i=0; i<k; i++) {
        #pragma omp parallel for collapse(2) default(none) shared(next, cur, N)
        for (y=0; y<N; y++) {
            for (x=0; x<N; x++) {
                const int xnext = (2*x+y) % N;
                const int ynext = (x + y) % N;
                next[ynext*N + xnext] = cur[x+y*N];
            }
        }
        /* Swap old and new */
        tmp = cur;
        cur = next;
        next = tmp;
    }
    img->bmap = cur;
    free(next);
}

void cat_map_interchange( PGM_image* img, int k )
{
    int i, x, y;
    const int N = img->width;
    unsigned char *cur = img->bmap;
    unsigned char *next = (unsigned char*)malloc( N*N*sizeof(unsigned char) );

    assert(next != NULL);
    assert(img->width == img->height);

    #pragma omp parallel for collapse(2) private(i) default(none) shared(next, cur, N, k)
    for (y=0; y<N; y++) {
        for (x=0; x<N; x++) {
            /* Compute the k-th iterate of pixel (x, y) */
            int xcur = x, ycur = y;
            for (i=0; i<k; i++) {
                const int xnext = (2*xcur+ycur) % N;
                const int ynext = (xcur + ycur) % N;
                xcur = xnext;
                ycur = ynext;
            }
            next[ycur*N+xcur] = cur[y*N+x];
        }
    }
    img->bmap = next;
    free(cur);
}

int main( int argc, char* argv[] )
{
    PGM_image img;
    int niter;
    double elapsed;
    const int NTESTS = 5; /* number of replications */

    if ( argc != 2 ) {
        fprintf(stderr, "Usage: %s niter < input > output\n", argv[0]);
        return EXIT_FAILURE;
    }
    niter = atoi(argv[1]);
    read_pgm(stdin, &img);

    if ( img.width != img.height ) {
        fprintf(stderr, "FATAL: width (%d) and height (%d) of the input image must be equal\n", img.width, img.height);
        return EXIT_FAILURE;
    }

    /**
     ** WITHOUT loop interchange
     **/
    elapsed = 0.0;
    for (int i=0; i<NTESTS; i++) {
        fprintf(stderr, "Run %d of %d\n", i+1, NTESTS);
        const double tstart = hpc_gettime();
        cat_map(&img, niter);
        elapsed += hpc_gettime() - tstart;
        if (i==0)
            write_pgm(stdout, &img, "produced by omp-cat-map.c");
    }
    elapsed /= NTESTS;

    fprintf(stderr, "\n=== Without loop interchange ===\n");
#if defined(_OPENMP)
    fprintf(stderr, "  OpenMP threads : %d\n", omp_get_max_threads());
#else
    fprintf(stderr, "  OpenMP disabled\n");
#endif
    fprintf(stderr, "      Iterations : %d\n", niter);
    fprintf(stderr, "    width,height : %d,%d\n", img.width, img.height);
    fprintf(stderr, "        Mops/sec : %f\n", 1.0e-6 * img.width * img.height * niter / elapsed);
    fprintf(stderr, "Elapsed time (s) : %f\n\n", elapsed);

    /**
     ** WITH loop interchange
     **/
    elapsed = 0.0;
    for (int i=0; i<NTESTS; i++) {
        fprintf(stderr, "Run %d of %d\n", i+1, NTESTS);
        const double tstart = hpc_gettime();
        cat_map_interchange(&img, niter);
        elapsed += hpc_gettime() - tstart;
    }
    elapsed /= NTESTS;

    fprintf(stderr, "\n=== With loop interchange ===\n");
#if defined(_OPENMP)
    fprintf(stderr, "  OpenMP threads : %d\n", omp_get_max_threads());
#else
    fprintf(stderr, "  OpenMP disabled\n");
#endif
    fprintf(stderr, "      Iterations : %d\n", niter);
    fprintf(stderr, "    width,height : %d,%d\n", img.width, img.height);
    fprintf(stderr, "        Mops/sec : %f\n", 1.0e-6 * img.width * img.height * niter / elapsed);
    fprintf(stderr, "Elapsed time (s) : %f\n\n", elapsed);

    free_pgm( &img );
    return EXIT_SUCCESS;
}
