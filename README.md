# linalg-rs - Linear Algebra library written in rust

## Linear algebra in Rust!

---

Parallelized using rayon with support for many common datatypes,
linalg-rs tries to make matrix operations easier for the user,
while still giving you as the user the performance you deserve.

Regular matrices have many features already ready, while
Sparse ones have most of them. Whenever you want to switch from
one to the other, just call `from_dense`, or `from_sparse` to
quickly and easily convert!

Need a feature? Please let me/us know!W

Even have custom declarative macros to create hashmap for your
sparse matrices!

## Examples

### Dens Matrices

```rust
use linalg_rs::{LinAlgFloats, Matrix};

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
use linalg_rs::{SparseMatrix, SparseMatrixData};

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

## Features

- [x] Easy to use!
- [x] Blazingly fast
- [x] Linear Algebra module fully functional on f32 and f64
- [x] Optimized matrix multiplication for both sparse and dense matrices
- [x] Easily able to convert between sparse and dense matrices
- [x] Serde support
- [x] Support for all signed numeric datatypes
- [x] Can be sent over threads
- [x] Sparse matrices
