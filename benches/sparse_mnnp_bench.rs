use std::collections::HashMap;
use sukker::{smd, SparseMatrix, SparseMatrixData};

use criterion::{black_box, criterion_group, criterion_main, Criterion};

// Benchmark for matrix multiplication
fn sparse_matmul_bench(c: &mut Criterion) {
    let indexes: SparseMatrixData<f64> = smd![
        ((0, 1), 2.0),
        ((1, 9), 4.0),
        ((1, 8), 6.0),
        ((9, 9), 8.0),
        ((6, 2), 8.0),
        ((2, 2), 8.0),
        ((1, 3), 8.0),
        ((8, 6), 8.0),
        ((9, 1), 8.0),
        ((2, 3), 8.0),
        ((7, 9), 8.0),
        ((7, 8), 8.0),
        ((3, 8), 8.0),
        ((0, 9), 8.0)
    ];

    let x = black_box(SparseMatrix::<f64>::new(indexes, (99, 100)));

    let indexes2: SparseMatrixData<f64> = smd![
        ((0, 0), 2.0),
        ((1, 0), 4.0),
        ((1, 1), 8.0),
        ((2, 1), 6.0),
        ((8, 1), 3.0),
        ((2, 5), 1.0),
        ((8, 1), 12.0),
        ((9, 8), 4.0),
        ((8, 2), 2.0),
        ((2, 4), 6.0),
        ((3, 9), 1.0),
        ((7, 0), 6.0),
        ((7, 0), 8.0),
        ((9, 7), 6.0)
    ];

    let y = black_box(SparseMatrix::<f64>::new(indexes2, (100, 99)));

    c.bench_function("MxN @ NxP sparse matmul", |b| {
        b.iter(|| x.matmul_sparse(&y).unwrap())
    });
}

criterion_group!(benches, sparse_matmul_bench);
criterion_main!(benches);
