# Questions and Answers

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
### What is the largest planet in our solar system?
#### Answer:
Jupiter.

---

## Question 4:
### Who wrote "To Kill a Mockingbird"?
#### Answer:
Harper Lee.

---

## Question 5:
### What is the chemical symbol for water?
#### Answer:
H2O.

---
