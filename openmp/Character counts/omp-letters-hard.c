#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <string.h>
#include <assert.h>
#include <omp.h>

#define ALPHA_SIZE 26

/**
 * Count occurrences of letters 'a'..'z' in `text`; uppercase
 * characters are transformed into lowercase, and all other symbols
 * are ignored. `text` must be zero-terminated. `hist` will be filled
 * with the computed counts. Returns the total number of letters
 * found.
 */
int make_hist( const char *text, unsigned int ** words , int thread_num, unsigned int* hist)
{
    int nlet = 0; /* total number of alphabetic characters processed */
    int *shared_nlet = (int*)calloc(thread_num, sizeof(int)*ALPHA_SIZE); /* total number of alphabetic characters processed by a specific thread */
    const size_t TEXT_LEN = strlen(text);
    int i, j;
    int my_start; /* index of first letter the threads are going to compute,
    its value is changed in the parallel section */
    int my_end;
    int my_id;
    /* Reset the words count */
    for (j=0; j<thread_num; j++) {
        words[j] = (unsigned int*)calloc(ALPHA_SIZE, sizeof(unsigned int));
    }

    #pragma omp parallel default(none) shared(text, words, shared_nlet, TEXT_LEN, thread_num) private(my_id, my_start, my_end, i)
{
    my_id = omp_get_thread_num();
    my_start = (TEXT_LEN * omp_get_thread_num())/omp_get_num_threads();
    my_end = (TEXT_LEN * (omp_get_thread_num() + 1))/omp_get_num_threads();
    /* Count occurrences */
    for (i=my_start; i<my_end; i++) {
        const char c = text[i];
        if (isalpha(c)) {
            words[my_id][ tolower(c) - 'a' ]++;
            shared_nlet[thread_num]++;
        }
    }
}

    for (i=0; i<ALPHA_SIZE; i++) {
        for (j=0; j<thread_num; j++) {
            hist[i]+=words[j][i];
        }            
    }

    for (j=0; j<thread_num; j++) {
        nlet+=shared_nlet[j];
    }
    
    free(shared_nlet);

    return nlet;
}

/**
 * Print frequencies
 */
void print_hist( unsigned int hist[ALPHA_SIZE] )
{
    int i;
    unsigned int nlet = 0;
    for (i=0; i<ALPHA_SIZE; i++) {
        nlet += hist[i];
    }
    for (i=0; i<ALPHA_SIZE; i++) {
        printf("%c : %8d (%6.2f%%)\n", 'a'+i, hist[i], 100.0*hist[i]/nlet);
    }
    printf("    %8d total\n", nlet);
}

int main( void )
{
    unsigned int ** words = (unsigned int**)malloc(num_threads*sizeof(unsigned int*));
    unsigned int * hist = (unsigned int*)calloc(ALPHA_SIZE, sizeof(unsigned int));
    const size_t size = 5*1024*1024; /* maximum text size: 5 MB */
    char *text = (char*)malloc(size); assert(text != NULL);
    const char* str_num_threads = getenv("OMP_NUM_THREADS");
    int num_threads;

    assert(str_num_threads!=NULL);
    num_threads = atol(str_num_threads);

    const size_t len = fread(text, 1, size-1, stdin);
    text[len] = '\0'; /* put a termination mark at the end of the text */
    const double tstart = omp_get_wtime();
    make_hist(text, words, num_threads, hist);
    const double elapsed = omp_get_wtime() - tstart;
    print_hist(hist);
    fprintf(stderr, "Elapsed time: %f\n", elapsed);
    free(text);
    free(words);
    return EXIT_SUCCESS;
}
