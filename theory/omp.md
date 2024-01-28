# OpenMP Q/A

## Question 1:
### In what sense is OpenMP a model based on _incremental parallelization_?
#### Answer:
It is in the sense of parallelization of some parts of the programs, one at the time.
[Here reference.](https://www.cise.ufl.edu/research/ParallelPatterns/glossary.htm#glossary:incremental-parallelism)


---

## Question 2:
### What are specific limitations of OpenMP?
#### Answer:
- OpenMP does not let you parallelize automatically, like a SIMD allowing compiler would.
- It does __not__ garauntee speed up. For example  parallelized parallel regions could be parallelized in an highly inefficient way or the overhead caused by syncronization of execution units could overrule the speed up gained by the parallelization.
- When data dependencies are present, _data races_ are not handled by anyone except the responsible programmer with directives provided by the omp libraries like _pragma_omp_critial_ or _pragma_omp_atomic_ or with any code restructurization that prevents data dependencies. In distribuited memory parallelization paradigms this does not happen.
---

## Question 3:
### What is oversubscription? Give some examples on scenarios where this thing may happen
#### Answer:
Oversubscription happens when there are more __logical threads__, which are executing parallel regions, than processor cores. 

This might happen with nested parallelism: let _p_ be the number of threads executing a parallel region _p1_ and let this be the ideal case where _p_ = number of physical cores of the CPU(So no Hyperthreading for example). <br>
 
 If there is another parallel region _p2_ defined inside the structured block which _p1_ parallelizes then each thread will create its pool of, now logical, threads which will execute _p2_.<br>
 It is obvious that the parallel regions _p2_ will not be all executed in parallel(except in the case where the pool size in _p2_=1) thus penalizing the performance. 

 ---
 ## Question 4:
 ### Can you explain how the _reduction(\<op>: \<variable>)_ works?
 #### Answer:
 Each thread gets a private copy of _variable_, which means each thread will see _variable_ initialized with neutral value in respect to _op_.<br>
 At the end of the structured block in which the reduction is decleared, the private value of _variable_ of each thread is going to be computed with _op_ operation and with the value of _variable_ that it had before entering the block on which the reduction has been called.

 ```
    ...
int a = 2;
#pragma omp parallel reduction(*:a) num_threads(3)
{
/* implicit initialization a = 1 */
a += 2;
}
/**/

 ```