# Parallelizing Loops Q/A

## Question 1:
### Refer to the following loop, how would you remove the data dependency so that the loop can be parallelized?
```
#define N some_big_number
double a[N];
const double f = 1.000001;
double val = 1.0;
int i;
for (i=0; i<N; i++) {
    a[i] = val;
    val = val * f
}
```
#### Answer:
Osservation: the loop modifies the vector `a` in the following way: `a[0]=val, a[1]=val*f, a[2]=val*f*f, ...`.

So `a[i]=val*(f^i)`.

```
...
#include <math.h>
...
/*This loop can be parallelized... */
for (i=0; i<N; i++) {
    a[i] = val * pow(f, i);
}
```

---

## Question 2:
### Refer to the following loop, how would you reconstruct the code so that one or more loops can be parallelized?
```
#define N some_big_number
int s[N] = {1, 1, ... 1};
double p[N] = { ... };
int i, j;
/* Assume that all s[i] are initially 1, and that all p[i] have been suitably initialized. Assume that function f() has no side effects */
for (i=0; i<N; i++) {
    if (s[i]) {
        for (j=0; j<N; j++) {
            if (s[j] && f(p[i], p[j])) {
                s[j] = 0;
            }
        }
    }
}
```
#### Answer:

Here, the outer loop is not parallelizable because it would create a _data race_ on array _s_.

But the inner loop, with the assumption that f has no side effects, can be parallelized. 

---

## Question 3:
### Refer to the following loop, how would you reconstruct the code so that one or more loops can be parallelized?
```
#define N 10000
double phi[2][N][N], maxdelta;
const double EPS = 1.0e-6;
int cur = 0, next = 1;
/* ...Initializations not shown... */
do {
    maxdelta = 0.0;
    for (int i=1; i<N-1; i++) {
        for (int j=1; j<N-1; j++) {
            phi[next][i][j] = (phi[cur][i+1][j] + phi[cur][i-1][j] + 
            phi[cur][i][j+1] + phi[cur][i][j-1]) / 4;
            const double delta = fabs(phi[next][i][j] - phi[cur][i][j]);
            if (delta > maxdelta) {
                maxdelta = delta;
            }
        }
    }
    /* exchange “cur” and “next” */
    const int tmp = cur; cur = next; next = tmp;

} while (maxdelta > EPS);
```
#### Answer:

This is embarassingly parallel pattern, we can parallelize both loops, the only thing problematic is the `maxdelta` variable  we have being written potentially by more than one thread.
To address this one solution would be to create `private` max delta variables for each thread and then compute a `reduce` operation with associative operator `max`.


For OpenMP a `collapse(2)` with fixed size chunks of statically scheduled iterations would be the best choice as there is no sign of load unbalancing between iterations.

---
