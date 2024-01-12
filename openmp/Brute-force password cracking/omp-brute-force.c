/****************************************************************************
 *
 * omp-brute-force.c - Brute-force password cracking
 *
 * Copyright (C) 2017--2022 by Moreno Marzolla <moreno.marzolla(at)unibo.it>
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 ****************************************************************************/

/***
% HPC - Brute-force password cracking
% Moreno Marzolla <moreno.marzolla@unibo.it>
% Last updated: 2022-10-18

![[DES cracker board](https://en.wikipedia.org/wiki/EFF_DES_cracker) developed in 1998 by the Electronic Frontier Foundation (EFF); this device can be used to brute-force a DES key. The original uploader was Matt Crypto at English Wikipedia. Later versions were uploaded by Ed g2s at en.wikipedia - CC BY 3.0 us, <https://commons.wikimedia.org/w/index.php?curid=2437815>](des-cracker.jpg)

The program [omp-brute-force.c](omp-brute-force.c) contains a 64-Byte
encrypted message stored in the array `enc[]`. The message has been
encrypted using the _XOR_ cryptographic algorithm, which applies the
"exclusive or" (xor) operator between a plaintext and the encryption
key. The _XOR_ algorithm is _not_ secure but on some special cases
(e.g., when the key has the same length of the plaintext, and the key
contains truly random bytes); however, it is certainly "good enough"
for this exercise.

_XOR_ is a _symmetric_ encryption algorithm, meaning that the same key
must be used for encrypting and decrypting a message. Indeed, the
exact same algorithm can be used to encrypt or decrypt a message, as
shown below.

The program contains a function `xorcrypt(in, out, n, key, keylen)`
that can be used to encrypt or decrypt a message with a given key. To
encrypt a message, then `in` points to the plaintext and `out` points
to a memory buffer that will contain the ciphertext. To decrypt a
message, then `in` points to the ciphertext and `out` points to a
memory buffer that will contain the plaintext.

The parameters are as follows:

- `in` points to the source message. This buffer does not need to be
  zero-terminated since it may contain arbitrary bytes;

- `out` points to a memory buffer of at least $n$ Bytes (the same
  length of the source message), that must be allocated by the caller.
  At the end, this buffer contains the source message that has been
  encrypted/decrypted with the encryption key;

- `n` is the length, in Bytes, of the source message;

- `key` points to the encryption/decryption key. The key is
  interpreted as a sequence of arbitrary bytes, and therefore does not
  need to be zero-terminated

- `keylen` is the length of the encryption/decryption key.

The _XOR_ algorithm will decrypt any message with any provided key; if
the key is not correct, the decrypted message will not make any
sense. For this exercise the plaintext is a zero-terminated ASCII
string that begins with `0123456789`.

The encryption key that has been used in this program is a sequence of
8 ASCII numeric characters; therefore, the key is a string between
`"00000000"` and `"99999999"`. Write a program to brute-force the key
using OpenMP. The program should try every key until a valid message
is eventually found, i.e., a message that begins with `0123456789`. At
the end, the program must print the plaintext, which is a famous quote
from [an old film](https://en.wikipedia.org/wiki/WarGames).

As you can see, the main loop of this program is not in a form that
can be parallelized with the `omp for` construct (why?).  Therefore,
you must use the `omp parallel` construct to partition the key space
among threads. Remember that `omp parallel` applies to a _structured
block_ with a single entry and a single exit point. Therefore, the
thread who finds the correct key can not exit the parallel region
using `return`, `break` or `goto` (the compiler should raise a
compile-time error). However, we certainly want to terminate the
program as soon as the key has been found.  Therefore, we need to
think of a clean mechanism to exit the loop when the correct key has
been found.

Compile with:

        gcc -std=c99 -Wall -Wpedantic -fopenmp omp-brute-force.c -o omp-brute-force

Run with:

        OMP_NUM_THREADS=2 ./omp-brute-force

**Note**: the execution time of the parallel program might change
irregularly depending on the number $P$ of OpenMP threads. Why?

## Files

You can use the `wget` command to easily transfer the files on the lab
server:

        wget https://www.moreno.marzolla.name/teaching/HPC/handouts/omp-brute-force.c

- [omp-brute-force.c](omp-brute-force.c)

***/

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
    char* out = (char*)malloc(msglen); /* where to put the decrypted message */
    assert(out != NULL);

    for (k=0; k < n && !found; k++) {
        snprintf(key, KEY_LEN+1, "%08d", k);
        xorcrypt(enc, out, msglen, key, KEY_LEN);
        /* `out` contains the decrypted text; if the key is not
           corret, `out` will contain garbage */
        if ( 0 == memcmp(out, check, CHECK_LEN) ) {
            printf("Key found: %s\n", key);
            printf("Decrypted message: \"%s\"\n", out);
            found = 1;
        }
    }
    assert(found); /* ensure that we did found the key */
    free(out);
    return EXIT_SUCCESS;
}
