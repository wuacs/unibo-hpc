#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <omp.h>

/* Fill v[] with the constant 1 */
void fill(int* v, int n)
{
    int i;
    printf("Initializing %d elements...\n", n);
    for (i=0; i<n; i++) {
        v[i] = 1;
    }
}

void check(int *s, int n)
{
    int i;
    for (i=0; i<n; i++) {
        if ( s[i] != i+1 ) {
            printf("Check failed: expected s[%d]==%d, got %d\n", i, i+1, s[i]);
            abort();
        }
    }
    printf("Check ok!\n");
}

/* Compute the inclusive scan of the n-elements array v[], and store
   the result in s[]. The caller is responsible for allocating s with
   n elements */
void inclusive_scan(int *v, int n, int *s, int num_threads)
{
    int i, j, k;
    int my_start, my_end, my_id;
    int lead_elements[num_threads];

    printf("Scanning %d elements...\n", n);
    /* degenerate case of empty array: do nothing */
    if ( n == 0 )
        return;

    s[0] = v[0];
    #pragma omp parallel default(none) private(my_id, my_start, my_end, i) shared(num_threads, s, v, n, lead_elements)
{
    my_id = omp_get_thread_num();
    my_start = (n*my_id)/num_threads;
    my_end = (n*(1+my_id))/num_threads;
    s[my_start] = v[my_start];
    for (i=my_start+1; i<my_end; i++) {
        s[i] = s[i-1] + v[i];
    }
    lead_elements[my_id] = s[my_end-1];
}
    
    for (j=1; j<num_threads; j++) {
        lead_elements[j] = lead_elements[j-1] + lead_elements[j];
    }
    
    #pragma omp parallel default(none) private(my_id, my_start, my_end, k) shared(num_threads,s, n, lead_elements) 
{   
    my_id = omp_get_thread_num();
    if (my_id !=0 ) {
        my_start = (n*(my_id))/num_threads;
        my_end = (n*(1+my_id))/num_threads;
        for (k = my_start; k<my_end; k++) {
            s[k]+=lead_elements[my_id-1];
        }
    }
}
} 

int main( int argc, char *argv[] )
{
    int n = 1000000;
    int *v, *s;
    int num_threads;
    char * str_num_threads = getenv("OMP_NUM_THREADS");

    assert(str_num_threads!=NULL);
    num_threads = atoi(str_num_threads);

    if ( argc > 2 ) {
        fprintf(stderr, "Usage: %s [n]\n", argv[0]);
        return EXIT_FAILURE;
    }

    if ( argc == 2 ) {
        n = atoi(argv[1]);
    }

    if (n > (1<<29)) {
        fprintf(stderr, "FATAL: array too large\n");
        return EXIT_FAILURE;
    }

    v = (int*)malloc(n*sizeof(int)); assert(v != NULL);
    s = (int*)malloc(n*sizeof(int)); assert(s != NULL);
    fill(v, n);
    inclusive_scan(v, n, s, num_threads);
    check(s, n);
    free(v);
    free(s);
    return EXIT_SUCCESS;
}
