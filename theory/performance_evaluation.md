# Performance Q/A

## Question 1:
### Define Amdahl's law, what are its limitations?
#### Answer:

Amdahl's law says that the speed up, obtainable by the parallezation of some parts of some program, is bounded by the percentage of serial activity a program executes.

$$Speedup(p)=\frac{T_{serial}}{T_{parallel}}=\frac{T_{serial}}{\alpha T_{serial}+(1-\alpha)\frac{T_{serial}}{p}}$$

$$\lim_{p \to \infty}{Speedup(p)}=\frac{T_{serial}}{\alpha T_{serial}}=\frac{1}{\alpha}$$

__Important Observations:__

- This result holds only if we have `perfectly` parallelizable parts. It does __not__ take into account `overhead`, `bottlenecks`, `communication costs` which should penalize the speed up as the more executing units we have the more communication has to be done.

- It is a very limited point of view to judge that a program's speedup is solely bounded by the fraction of time it spends in serial execution. This becomes clear as lots of high performance computing is done on very large `work domains`, which is parallelized, and little time is spent on serial instructions. This gave an idea to Lars Gustaffson which then elaborated the `Gustaffson Law` and expanded on the concept of Efficiency.
---

## Question 2:
### Define what is Weak Scaling Efficiency and give an example on Matrix product.
#### Answer:

`Weak Scaling Efficiency` (WSE) is a measure which gives us how much a program's time execution scales well with increasing problem size. <br>
Ideally it would remain fixed at `1` if incrementing the number of processes and the problem size by a fixed rate would make the computation take always the same time. <br>

Since it is obsolete to only talk about times(because we are updating processes and problem size) the definition of speed up used by Amdahl's law has to be re-adapted.

Instead of calling it `Speedup 2.0` we are just going to call it `WSE` referecing the metric which created this mapping.

$$WSE(p)=\frac{T_1}{T_p}$$

Where $T_1$ is the time the program needs to execute a `base work load` as problem size using `1` execution unit.<br>
$T_p$ is the time the program needs to execute an appropriate calculated problem size using `p` execution units, the problem size is calculated so to keep a `base work load` number of parallel instructions executed for each of the `p` processors used.

__Matrix product example__

Let's say we need product of matrix `A`(mxn) and `B`(kxm) and suppose we do a coarse grained division of domain since no load inbalancing can incur in dot products. <br>
Namely, we want to calculate how much we do need to increase `m` in order to keep the time equal to the base case i.e. $T_1$.

Hypothesize that the algorithm takes $O(n^3)$ so ideally each thread is going to take $O((n^3)/p)$ time/work to finish.

Let's say we choose `base work load` to be `1024` which is `m` value.<br>
$T_1=n^3=1024^3$
$$\frac{m^3}{p}=1024^3\rightarrow m^3=1024^3 \cdot p\rightarrow m=\sqrt[3]{p}\cdot 1024$$

And now we have the formula for calculating the new `m` to have each thread do "`1024^3`" computation.
 
---
