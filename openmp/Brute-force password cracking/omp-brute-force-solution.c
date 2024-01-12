#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <omp.h>

/* Decrypt `enc` of length `n` bytes into buffer `dec` using `key` of
   length `keylen`. The encrypted message, decrypted messages and key
   are treated as binary blobs; hence, they do not need to be
   zero-terminated.

   Do not modify this function. */
void xorcrypt(const char* in, char* out, int n, const char* key, int keylen)
{
    for (int i=0; i<n; i++) {
        out[i] = in[i] ^ key[i % keylen];
    }
}


int main( int argc, char *argv[] )
{
    const int KEY_LEN = 8;
    /* encrypted message */
    const char enc[] = {
        4, 1, 0, 1, 0, 1, 4, 1,
        12, 9, 115, 18, 71, 64, 64, 87,
        90, 87, 87, 18, 83, 85, 95, 83,
        26, 16, 102, 90, 81, 20, 93, 88,
        88, 73, 18, 69, 93, 90, 92, 95,
        90, 87, 18, 95, 91, 66, 87, 22,
        93, 67, 18, 92, 91, 64, 18, 66,
        91, 16, 66, 94, 85, 77, 28, 54
    };
    const int msglen = sizeof(enc);
    const char check[] = "0123456789"; /* the correctly decrypted message starts with these characters */
    const int CHECK_LEN = strlen(check);
    const int n = 100000000; /* number of possible keys */
    char key[KEY_LEN+1]; /* sprintf will output the trailing \0, so we need one byte more for the key */
    int k; /* numeric value of the key to try */
    int found = 0;
    int my_start; /* Included */
    int my_end; /* Excluded */
    char* out;
    #pragma omp parallel default(none) private(out, my_end, my_start, k, key) \
    shared(found, KEY_LEN, enc, msglen, n, check, CHECK_LEN)
{

    my_start = (n * omp_get_thread_num())/omp_get_num_threads();
    my_end = (n * (omp_get_thread_num()+1))/omp_get_num_threads();
    out = (char*)malloc(msglen); /* where to put the decrypted message */
    assert(out != NULL);
    for (k=my_start; k < my_end && !found; k++) {
        snprintf(key, KEY_LEN+1, "%08d", k);
        xorcrypt(enc, out, msglen, key, KEY_LEN);
        /* `out` contains the decrypted text; if the key is not
           corret, `out` will contain garbage */
        if ( 0 == memcmp(out, check, CHECK_LEN) ) {
            printf("Key found: %s\n", key);
            printf("Decrypted message: \"%s\"\n", out);
            #pragma omp critical
            found = 1;
        }
    }
    free(out);
}
    assert(found); /* ensure that we did found the key */
    return EXIT_SUCCESS;
}
