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

## Question 4:
### Consider the scenario where you need to send a column of a matrix from a MPI processor to another MPI processor, what are the solutions you can describe?

### Answer:

First solution is using a series of `MPI_Send` with datatype of sending element set to the type of the matrix elements. This is highly inefficient because of all the overhead costs of the communication of setting and unsetting all the `MPI_Send`(or `MPI_ISend`).

Assuming the matrix is n x m, a second solution is writing all elements in a temporary array of size m and then using an `MPI_Send` to send the array to the other MPI processor. 
This is better than the first solution but still inefficient due to array copying.

Third solution uses `Custom Datatypes`: in the particular case we can use `MPI_Type_vector()` to create a Datatype of `strided` elements of a particular type.<br>
If we were in the case of a matrix of structs we could create a Datatype for the struct using `MPI_Type_create_struct()` and then using this datatype in our `MPI_Type_vector` to accomplish the same result we would have done with any standard datatype.

---

## Question 5:
### What are the main differences between instructions like MPI_Scatter(), MPI_Receive and their counterparts MPI_Scatterv(), MPI_Gatherv()?

### Answer:

Instruction like `MPI_Scatter()` and `MPI_Receive()` are collective communication instructions.
Let's look at `MPI_Scatter()` but almost everything is applicable to the other instruction aswell.
```
/* define sendbug and recvbuf */
int sendcnt = 3; /* how many items are sent to each process */
int recvcnt = 3; /* how many items are received by each process */
int src = 1; /* process 1 contains the message to be scattered */
MPI_Scatter(sendbuf, sendcnt, MPI_INT, recvbuf, recvcnt, MPI_INT, src, MPI_COMM_WORLD);
```

Let's say `sendbuf` is an array of 11 elements and we are sending to 3 processes(including the sender).

So the buffer data is copied in `recvbuf` as follows: <br>
Processor 0: sendbuf[0 to 3] <br>
Processor 1: sendbuf[3 to 6] <br>
Processor 2: sendbuf[6 to 9] <br>

Notice that we __cannot__ send sendbuf[10] to no MPI_Process because if we sent 4 elements instead of 3 we would send elements on which we do not have access to!

To address this situation, and to allow MPI programmers to define their own division of elements, `MPI_Scatterv` was made.

Let's say we want to send the last element, sendbuf[10], to Processor 2:

```
int sendcnts[] = {3, 3, 4};
int displs[] = {0, 3, 6};
int recvcnt = 3;
int src = 1;
MPI_Scatterv(sendbuf, sendcnts, displs, MPI_INT, recvbuf, int recvcnt, MPI_INT, src, MPI_COMM_WORLD);
```

