

`The file omp-letters-efficient.c is a solution which uses OpenMP 4.5 newly added operations i.e. reduction(+:hist[:ALPHA_SIZE]) which the compiler interprets as ALPHA_SIZE reduction clauses i.e. reduction(+:hist[0]), reduction(+:hist[1]), ..., reduction(+:hist[ALPHA_SIZE-1])`

`The hist[:ALPHA_SIZE] tells the compiler to create reduction clauses from first position of hist array to the right number(excluded.)`