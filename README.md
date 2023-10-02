# Sukker - Linear Algebra library written in rust

![Build Status](https://github.com/Jafagervik/sukker/actions/workflows/test.yml/badge.svg)
[![Documentation](https://docs.rs/sukker/badge.svg)](https://docs.rs/sukker/)
[![Crates.io](https://img.shields.io/crates/v/sukker.svg)](https://crates.io/crates/sukker)
[![Coverage Status](https://codecov.io/gh/Jafagervik/sukker/branch/master/graph/badge.svg)](https://codecov.io/gh/Jafagervik/sukker)
![Maintenance](https://img.shields.io/badge/maintenance-experimental-blue.svg)
![License](https://img.shields.io/crates/l/sukker)

Linear algebra in Rust!
---

Parallelized using rayon with support for many common datatypes,
sukker tries to make matrix operations easier for the user, 
while still giving you as the user the performance you deserve.

Regular matrices have many features already ready, while 
Sparse ones have most of them. Whenever you want to switch from 
one to the other, just call `from_dense`, or `from_sparse` to
quickly and easily convert!

Need a feature? Please let me/us know!

Even have custom declarative macros to create hashmap for your
sparse matrices!

## Examples


### Dens Matrices 

```rust 
use sukker::{LinAlgFloats, Matrix};

fn main() {
    let a = Matrix::<f32>::randomize((8, 56));
    let b = Matrix::<f32>::randomize((56, 8));

    let c = a.matmul(&b).unwrap();

    let res = c.sin().exp(3).unwrap().pow(2).add_val(4.0).abs();

    // To print this beautiful matrix:
    res.print(5);
}
```

### Sparse Matrices 


```rust 
use std::collections::HashMap;
use sukker::{SparseMatrix, SparseMatrixData};

fn main() {
    let indexes: SparseMatrixData<f64> = smd![
        ((0, 1), 2.0), 
        ((1, 0), 4.0), 
        ((2, 3), 6.0), 
        ((3, 3), 8.0)
    ];

    let sparse = SparseMatrix::<f64>::new(indexes, (4, 4));

    sparse.print(3);
}
```

More examples can be found [here](/examples/)


## Documentation
Full API documentation can be found [here](https://docs.rs/sukker/latest/sukker/).

## Features 
- [X] Easy to use!
- [X] Blazingly fast
- [X] Linear Algebra module fully functional on f32 and f64
- [X] Optimized matrix multiplication for both sparse and dense matrices 
- [X] Easily able to convert between sparse and dense matrices
- [X] Serde support 
- [X] Support for all signed numeric datatypes 
- [X] Can be sent over threads
- [X] Sparse matrices 

