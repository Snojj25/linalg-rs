use sukker::Matrix;

use criterion::{black_box, criterion_group, criterion_main, Criterion};

// Benchmark for matrix multiplication
fn blocked_matmul_bench(c: &mut Criterion) {
    let x = black_box(Matrix::<f64>::randomize((100, 100)));
    let y = black_box(Matrix::<f64>::randomize((100, 100)));

    c.bench_function("NxN @ NxN dense matmul", |b| {
        b.iter(|| x.matmul(&y).unwrap())
    });
}

criterion_group!(benches, blocked_matmul_bench);
criterion_main!(benches);
