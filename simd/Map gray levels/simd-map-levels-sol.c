/* The following #define is required to make posix_memalign() visible */
#define _XOPEN_SOURCE 600

#include "hpc.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

typedef int v4i __attribute__((vector_size(16)));
#define VLEN (sizeof(v4i)/sizeof(int))

typedef struct {
    int width;   /* Padded width of the image (in pixels); this is a multiple of VLEN */
    int true_width; /* True width of the image (in pixels) */
    int height;  /* Height of the image (in pixels) */
    int maxgrey; /* Don't care (used only by the PGM read/write routines) */
    int *bmap;   /* buffer of width*height bytes; each element represents the gray level of a pixel (0-255) */
} PGM_image;

const int BLACK = 0;
const int WHITE = 255;

/**
 * Read a PGM file from file `f`. Warning: this function is not
 * robust: it may fail on legal PGM images, and may crash on invalid
 * files since no proper error checking is done. The image width is
 * padded to the next integer multiple of 'VLEN`.
 */
void read_pgm( FILE *f, PGM_image* img )
{
    char buf[1024];
    const size_t BUFSIZE = sizeof(buf);
    char *s;

    assert(f != NULL);
    assert(img != NULL);

    /* Get the file type (must be "P5") */
    s = fgets(buf, BUFSIZE, f);
    if (0 != strcmp(s, "P5\n")) {
        fprintf(stderr, "Wrong file type \"%s\", expected \"P5\"\n", buf);
        exit(EXIT_FAILURE);
    }
    /* Get any comment and ignore it; does not work if there are
       leading spaces in the comment line */
    do {
        s = fgets(buf, BUFSIZE, f);
    } while (s[0] == '#');
    /* Get width, height */
    sscanf(s, "%d %d", &(img->true_width), &(img->height));
    /* Set `img->width` as the next integer multiple of `VLEN`
       greater than or equal to `img->true_width` */
    img->width = ((img->true_width + VLEN - 1) / VLEN) * VLEN;
    /* get maxgrey; must be less than or equal to 255 */
    s = fgets(buf, BUFSIZE, f);
    sscanf(s, "%d", &(img->maxgrey));
    if ( img->maxgrey > 255 ) {
        fprintf(stderr, "FATAL: maxgray=%d, expected <= 255\n", img->maxgrey);
        exit(EXIT_FAILURE);
    }
    /* The pointer `img->bmap` must be properly aligned to allow SIMD
       instructions, because the compiler emits SIMD instructions for
       aligned load/stores only. */
    int ret = posix_memalign((void**)&(img->bmap), __BIGGEST_ALIGNMENT__, (img->width)*(img->height)*sizeof(int));
    assert(0 == ret);
    assert(img->bmap != NULL);
    /* Get the binary data from the file */
    for (int i=0; i<img->height; i++) {
        for (int j=0; j<img->width; j++) {
            unsigned char c = WHITE;
            if (j < img->true_width) {
                const int nread = fscanf(f, "%c", &c);
                assert(nread == 1);
            }
            *(img->bmap + i*img->width + j) = c;
        }
    }
}

/**
 * Write the image `img` to file `f`; if not `NULL`, use the string
 * `comment` as metadata.
 */
void write_pgm( FILE *f, const PGM_image* img, const char *comment )
{
    assert(f != NULL);
    assert(img != NULL);

    fprintf(f, "P5\n");
    fprintf(f, "# %s\n", comment != NULL ? comment : "");
    fprintf(f, "%d %d\n", img->true_width, img->height);
    fprintf(f, "%d\n", img->maxgrey);
    for (int i=0; i<img->height; i++) {
        for (int j=0; j<img->true_width; j++) {
            fprintf(f, "%c", *(img->bmap + i*img->width + j));
        }
    }
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
    img->width = img->true_width = img->height = img->maxgrey = -1;
}

/*
 * Map the gray range [low, high] to [0, 255].
 */
void map_levels( PGM_image* img, int low, int high )
{   
    const int width = img->width;
    const int height = img->height;
    int *bmap = img->bmap;
    for (int i=0; i<height; i++) {
        for (int j=0; j<width; j+=VLEN) {
            v4i *pixels = (v4i*)(bmap + i*width + j);
            //const v4i mask_black = (*pixels < low);
            const v4i mask_white = (*pixels > high);
            const v4i mask_map = (*pixels >= low) & (*pixels <= high);
            *pixels = ( (mask_white & WHITE) | 
                        (mask_map & (255 * (*pixels - low) / (high - low))) ); 
        }
    }
}

int main( int argc, char* argv[] )
{
    PGM_image bmap;

    if ( argc != 3 ) {
        fprintf(stderr, "Usage: %s low high < in.pgm > out.pgm\n", argv[0]);
        return EXIT_FAILURE;
    }
    const int low = atoi(argv[1]);
    const int high = atoi(argv[2]);
    if (low < 0 || low > 255) {
        fprintf(stderr, "FATAL: low=%d out of range\n", low);
        return EXIT_FAILURE;
    }
    if (high < 0 || high > 255 || high <= low) {
        fprintf(stderr, "FATAL: high=%d out of range\n", high);
        return EXIT_FAILURE;
    }
    read_pgm(stdin, &bmap);
    const double tstart = hpc_gettime();
    map_levels(&bmap, low, high);
    const double elapsed = hpc_gettime() - tstart;
    write_pgm(stdout, &bmap, "produced by simd-map-levels-sol.c");
    fprintf(stderr, "Executon time (s): %f (%f Mops/s)\n", elapsed, (1e-6) * bmap.width * bmap.height / elapsed);
    free_pgm(&bmap);
    return EXIT_SUCCESS;
}
