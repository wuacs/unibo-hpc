# MPI Q/A

## Question 1:
### On which computational paradigm does MPI stand? And what, if any, differences has with SIMD paradigm?

#### Answer:

SPMD(Single Program Multiple Data) is the paradigm used by MPI. 
It uses multiple processors which at the same time can differ on the point of the control flow they are at. <br>
This opposes SIMD paradigm where all processors execute the same instructions, at the same time. <br>
It is important to note that one does not exclude the other: it's possible that in a MPI application one uses `vector processing` SIMD instructions or invokes GPU processing which relies on SIMD-like processors.

---

## Question 2:
### Describe advantages and disadvantages of Distribuited Memory Systems.

### Answer:

- Adavantages:
    - No data races
    - Forces the programmer to think of Locality(because he needs to handle alone distribution of domain's memory addresses).
- Disadvantages:
    - Possible communication errors.
    - Code more complex and bigger.

---

## Question 2:
### Describe advantages and disadvantages of Distribuited Memory Systems.

### Answer:

- Adavantages:
    - No data races
    - Forces the programmer to think of Locality(because he needs to handle alone distribution of domain's memory addresses).
- Disadvantages:
    - Possible communication errors.
    - Code more complex and bigger.

---

## Question 3:
### Explain in what do _point-to-point_ and _collective_ MPI communications differ.

### Answer:

Point-to-point transfer data from one to another processor: for example `MPI_Send` sends one or more `Datatypes` to another processor specified by an integer.
On the other hand, collective instructions such as `MPI_Scatter`, are normally meant to be executed by __ALL__ MPI processors and are to be preferred over Point-to-Point both could be used. The `MPI_Scatter` instruction is executed with `sender` role by only one processor(specified in `src` param.) and any other processor, when executing `MPI_Scatter` will act as `receiver`.

`MPI_Scatter` is blocking in the same way the `MPI_Send` is: the control flow of the `src` processor will resume if and only if the buffer which contains the bytes to be sent can be overwritten without incurring in conflict with the MPI driver which is copying the data to be transferred to some internal register.
 

---