# CUDA Q/A

## Question 1:
### Can you explain why we do not have to take care of the device computing capabilities when programming CUDA? What happens, for example, when you create more threads than Streaming Multiprocessors?

#### Answer

The CUDA scheduler is smart enough to create pseudo-optimal scheduling and, if it cannot execute all thread blocks at the same time, it will create a queue for access to the SMs.

---

## Question 2:
### Explain what a CUDA warp is.


#### Answer

A CUDA warp is a set of 32 CUDA threads which are the smallest logical executing block our GPU will execute.<br>
It is __logical__ because we do not have control on what hardware component is mapped to a CUDA Thread and this is good because it abstracts the underlining hardware allowing for portable CUDA programs.
<br>
<br>
All threads in a warp represent threads of a same logical CUDA block and those will be executed on one Streaming Multiprocessor with SIMD-like logic. 

---

## Question 3:
### Why is it important that threads in a warp execute the same instructions?

#### Answer

This is because, at any given point of time, if a warp is to be executed, the CUDA scheduler will look what is the Program Counter of every Streaming Processor and execute only the threads which share the same PC(decision on which PC it chooses is implementation based).<br>
<br>
So this means if we have 32 threads and each of them as a different Program Counter, at any point of time only 1 thread is executed and all the other are paused.

---

## Question 4:
### Why is it recommended to take care of spatial locality for CUDA warps(and so for the programmer of CUDA blocks)?

#### Answer

It is possible to maximize the throughput of the access on CUDA's memory if all threads of a warp access memory which is within the cache line of the CUDA global memory because the CUDA manager intercepts those requests and if it sees they are within the same cache line it will execute only one transaction of digging up this data(this in the ideal case where all warps access the same cache line but even two transactions are better than 32).

---

## Question 5:
### Why is efficient to have more threads per block than more blocks but less threads per block?

#### Answer

Because when all threads in a warp access data from memory, if you benefit from coalescens of memory access, the bus is going to be saturated and per unit of time you will be getting more data from the memory.

---
