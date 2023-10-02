use std::collections::HashMap;
use sukker::{SparseMatrix, SparseMatrixData};

fn main() {
    let sparse = SparseMatrix::<f32>::randomize_range(1.0, 7.5, 0.5, (8, 8));

    let sparse2 = SparseMatrix::randomize_range_like(4.0, 1000.0, &sparse);

    let result = match sparse.matmul(&sparse) {
        Some(v) => v,
        None => SparseMatrix::default(),
    };

    assert_eq!(result.shape(), (8, 8));

    assert_eq!(result.size(), 64);

    sparse.print(3);
}
