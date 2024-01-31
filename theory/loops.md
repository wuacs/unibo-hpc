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

## Question 4:
### Can you give classical classification of loop dependencies?

#### Answer:

We say there are generally three types of dependencies: `flow/true dependency`, `Anti dependency` and `output dependency`.

- A `flow dependency` is found when iteration _i_ of the loop writes on a memory address read by iteration _i+1_. Examples of this are reductions and Induction varibles.
    - `Reduction example`: 
    ```
    int a[] = {.....}
    int x = 0;
    for (int i=0; i<n; i++) {
        x+=a[i];
    }
    ```
    Can be rewritten as:
    ```
    int x = 0;
    int xs[SIZEOFTHREADS]; /* NEW */
    /* this for is parallelizable */
    for (int i=0; i<n; i++) {
        xs[my_id]+=a[my_id];
    }
    /* is to be done serially */
    for (int j=0; j<SIZEOFTHREADS; j++) {
        x+=xs[j];
    }
    ```
    - `Induction var. example`:
    ```
    j = 5;
    for (i = 1; i < n; i++) {
        j += 2;
        a[i] = f(j);
    }
    ```
    Can be rewritten as:
    ```
    for (i = 1; i < n; i++) {
        a[i] = f(5+2*i);
    }
    ```
- An `anti-dependence` is found when iteration _i_ of the loop reads on a memory address written by iteration _i+1_. Example:
    - Example:
    ```
    for (i = 0; i < n-1; i++)
        a[i] = a[i+1] + f(i);
    ``` 
    Solution consists in creating a copy of a.
    ```
    a_copy = a.copy();
    for (i = 0; i< n-1; i++)
        a[i] = a_copy[i+1] + f(i)
    ```
- An `output dependency` is found when iteration _i_ of the loop writes on a memory address written by iteration _i+1_.
    - Example:
    ```
    for (i = 0; i < n; i++) {
        tmp = A[i]; /* tmp is writeen by all iterations */
        A[i] = B[i];
        B[i] = tmp;
    }
    ```
    Solution is to make tmp private(with OpenMP using the _private_ directive):
    ```
    for (i = 0; i < n; i++) {
        int tmp = A[i];
        A[i] = B[i];
        B[i] = tmp;
    }
    ```
---

## Question 5:
### Refer to snippet below. How would you use an Even/Odd parallelization technique(with task parallelization). 
```
for (i = 2; i < n; i++)
    a[i] = a[i-2] + x;
```
#### Answer:

Observation: purpose of even/odd parallelization is to split iterations between iterations using `even` and `odd` indexes. So, in the example i = 2, i = 4, i = 6...can be given to a loop and i = 3, i = 5, i = 7... can be given to another one.
With OpenMP we can then parallelized those loops with task parallelization.
```
/* create parallel region */
#pragma omp single {
    #pragma omp task
    for (i = 2; i < n; i+=2)
        a[i] = a[i-2] + x;
}
#pragma omp single {
    #pragma omp task
    for (i = 3; i < n; i+=2)
        a[i] = a[i-2] + x;
}

```

## Question 6:
### Refer to the following loop, how would you reconstruct the code so that one or more loops can be parallelized? 
```
for ( i = 0; i < n-1; i++ )
    a[i] = a[i+1] + b[i] * c[i];    
```
#### Answer:

Observation: there is an `anti-dependency` on array `a`.

```
a_copy = a.copy();
for ( i = 0; i < n-1; i++ )
    a[i] = a_copy[i+1] + b[i] * c[i];
```

---

## Question 7:
### Refer to the following loop, how would you reconstruct the code so that one or more loops can be parallelized? 
```
for ( i = 1; i < n; i++ )
    a[i] = a[i-1] + b[i] * c[i];
/* a[0] = ~
a[1] = a[0] + ...
a[2] = a[0] + b[1]*c[1] + b[2]*c[2]
*/
```
#### Hint:

a[0] = ~ <br>
a[1] = a[0] + b[1]*c[1] <br>
a[2] = a[0] + b[1]*c[1] + b[2]*c[2] <br>
...

#### Answer:

Observation: there is an `flow dependency` on array `a`. Solution works only with OpenMP 5.0 or newer.

```
bc[n];
bperc = 0;

#pragma omp parallel for reduction(inscan, +:bperc)
for (i = 1; i < n; i++) {
    bperc += b[i]*c[i];
    #pragma omp scan inclusive(bperc)
    bc[i] = bperc;   
}

#pragma omp parallel for shared(a, bc, n)
for ( i = 1; i < n; i++ )
    a[i] = a[0] + bc[i];
```
---

## Question 8:
### Refer to the following loop, how would you reconstruct the code so that one or more loops can be parallelized? 
```
t = 1;
for ( i = 0; i < n-1; i++ ) {
    a[i] = a[i+1] + b[i] * c[i];
    t = t * a[i];
}   
```
#### Answer:

Observation: there is an `flow dependency` on array `a` and a `output dependency` on variable `t`.

Array `a` is changed based on `b` and `c` and also based on `a` initial values.<br>
So if we created a copy of `a` we could parallelize `a` updates.

With regards to `t` we simply observe that `t` is not used in the `for` iterations but simply calculated.<br>
When exiting the loop `t`'s expression is: `t=pow(t, n-1)*reduction(a)`.<br>
BUT, since `t` initial value is `1`, the expression simplifies into: `t=reduction(a)`.

In regard to OpenMP we can use clause `reduction(*:x)` to get `a[0]*a[1]*...*a[n-1]` written in a variable `x`.

```
t = 0;
a_copy[] = a.copy(); /* Pseudo copy function */

#pragma omp parallel for reduction(*:t)
for ( i = 0; i < n-1; i++) {
    t *= a[i];
}

for ( i = 0; i < n-1; i++ ) {
    a[i] = a_copy[i+1] + b[i] * c[i];
}   
```

---