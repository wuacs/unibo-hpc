# MPI Q/A

## Question 1:
### On which computational paradigm does MPI stand? And what, if any, differences has with SIMD paradigm?

#### Answer:

SPMD(Single Program Multiple Data) is the paradigm used by MPI. 
It uses multiple processors which at the same time can differ on the point of the control flow they are at. <br>
This opposes SIMD paradigm where all processors execute the same instructions, at the same time. <br>
It is important to note that one does not exclude the other: it's possible that in a MPI application one uses `vector processing` SIMD instructions or invokes GPU processing which relies on SIMD-like processors.

---
