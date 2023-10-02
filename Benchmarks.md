# Matrix Multiplication Benchmarks

All tests are done using cargo bench with criterion, no changes made, just usingblackbox and so on.

All tests are ran on IEEE754 double precision floating point numbers.

Sizes used are as following:
- M = 999
- N = 1000
- P = 999


## Dense Matrices

### NxN @ NxN 


### MxN @ NxP


## Sparse Matrices 


### NxN @ NxN 
Around 24.7ms

### MxN @ NxP
- Around 22.1ms


## Target specific matmuls

Not yet implemented, but X86-64 with AVX instrctions are soon to be implemented
