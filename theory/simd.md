# SIMD Q/A

## Question 1:
### See the snippet below. If we used SIMD instructions to optimize this loop and found a Superlinear Speed up what could be the cause(s)?
```
for (i=0; i<n; i++) {
    Compute(i)
}
``` 
#### Answer:
Possible SIMD implementation could be:
```
for (i=0; i<n-SIMD_REGISTER_SIZE+1; i+=SIMD_REGISTER_SIZE) {
    SIMDCOMPUTE(i+SIMD_REGISTER_SIZE);
}
/* Handle possible leftovers... */
``` 

The superlinear speed up, apart from errors in measuring, could be caused by:
- Less Fetch-Decode-Execute cycles of the CPU since we are doing n/SIMD_REGISTER_SIZE instructions instead of n.
- Less overhead cause by for increment and test since we are doing less iterations.

---
## Question 2:
### Refer to the code snippet proposed in question 1, how would you address the possibility of domain size not being multiple of the SIMD vector?
#### Answer:
- Padding: we could just extend the domain size with dummy values if this doesn't modify the behaviour of the program.
- Manual handling(scalar operations): we could leave to the main thread the work of handling alone the rest of the domain elements.
---
## Question 3:
### Refer to the following snippet, why do you think GCC 7.5.0 does __not__ autovectorize this loop?
```
float vsum(float *v, int n)
{
    float s = 0.0; int i;
    for (i=0; i<n; i++) {
        s += v[i];
    }
    return s;
}
```

#### Answer:
This is because float sums are __not__ associative. <br>
Scalar code would do the following operations in this order: v[0]+v[1]+v[2]...v[n-1]. <br>
Let SIMD vector be of 4 floats with start value {0.0, 0.0, 0.0, 0.0} then the operations are {v[0]+v[4]+v[8]+..., v[1]+v[5]+..., v[2]+v[6]+v[10]+..., v[3]+v[7]+...} <br>
As you can see v[0] is now summed first with v[4] instead of v[1].

---

## Question 4:
### How can you, in C99, help the compiler autovectorize better?
#### Answer:
One problem potentially worth adressing is _pointer_aliasing_, using the _restrict_ keyword we can help the compiler optimize better.

---
## Question 5:
### Refer to the snippet below. What tecniques can we use to address this divergent loop execution?
```
int count(int x) {
    int cnt = 0;
    for (int i = 0; i < N; i++)
        cnt += (a[i] == x);
    return cnt;
}
```
#### Answer:
One technique we can use is called _selection and masking_ or _masking and blending_.<br>
It supposes the use of operators like `<` in order to get as output a SIMD vector with value `0` where evaluation of the operation is `False` otherwise `-1`(which is rapresented as all ones 111..1).

Regarding the snippet: 
```
int count(int x) {
    int cnt = 0;
    v4d vec_cnt = {0, 0, 0, 0}; /* Assume vector data type of 4 integers */
    v4d * a_vec = (v4d*)a;
    for (int i = 0; i < N - 3; i+=4) {
        v4d mask = *a_ved == x; /* This produces some permutation of length 4 of {0,-1} */
        vec_cnt += (*mask & *a_ved); 
        a_vec++;
    }
    cnt = vec_cnt[0] + vec_cnt[1] + vec_cnt[2] + vec_cnt[3];
    /* Handle possible leftovers */
    return cnt;
}
```

---

## Question 6:
### The instruction for GCC compiler defines a SIMD type of length of 32 bytes, and assuming a float is 4 bytes, there can be 8 floats in a single v8f variable.
```
typedef float v8f __attribute__((vector_size(32)));
```
#### Answer:
One technique we can use is called _selection and masking_ or _masking and blending_.<br>
It supposes the use of operators like `<` in order to get as output a SIMD vector with value `0` where evaluation of the operation is `False` otherwise `-1`(which is rapresented as all ones 111..1).

Regarding the snippet: 
```
int count(int x) {
    int cnt = 0;
    v4d vec_cnt = {0, 0, 0, 0}; /* Assume vector data type of 4 integers */
    v4d * a_vec = (v4d*)a;
    for (int i = 0; i < N - 3; i+=4) {
        v4d mask = *a_ved == x; /* This produces some permutation of length 4 of {0,-1} */
        vec_cnt += (*mask & *a_ved); 
        a_vec++;
    }
    cnt = vec_cnt[0] + vec_cnt[1] + vec_cnt[2] + vec_cnt[3];
    /* Handle possible leftovers */
    return cnt;
}
```

