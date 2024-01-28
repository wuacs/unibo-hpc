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
...

---


