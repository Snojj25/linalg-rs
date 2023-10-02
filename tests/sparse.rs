use std::collections::HashMap;
use sukker::{smd, SparseMatrix, SparseMatrixData};

#[test]
fn sparse_basic() {
    let sparse1 = SparseMatrix::<i32>::eye(3);
    let sparse2 = SparseMatrix::<i32>::eye(3);

    let res = sparse1.add(&sparse2).unwrap();

    assert_eq!(res.shape(), (3, 3));
    assert_eq!(res.get(0, 0).unwrap(), 2);
    assert_eq!(res.get(1, 1).unwrap(), 2);
    assert_eq!(res.get(2, 2).unwrap(), 2);
    assert_eq!(res.get(1, 2).unwrap(), 0);
}

#[test]
fn sparse_medium() {
    let mut indexes: SparseMatrixData<f64> = HashMap::new();

    indexes.insert((0, 1), 2.0);
    indexes.insert((1, 0), 4.0);
    indexes.insert((2, 3), 6.0);
    indexes.insert((3, 3), 8.0);

    let sparse = SparseMatrix::<f64>::new(indexes, (4, 4));

    let indexes2 = smd![((0, 0), 2.0), ((1, 0), 4.0), ((1, 1), 8.0), ((2, 3), 6.0)];

    let sparse2 = SparseMatrix::<f64>::new(indexes2, (4, 4));

    let res = sparse.add(&sparse2).unwrap();

    assert_eq!(res.at(0, 0), 2.0);
    assert_eq!(res.at(0, 1), 2.0);
    assert_eq!(res.at(1, 0), 8.0);
    assert_eq!(res.at(1, 1), 8.0);
    assert_eq!(res.at(2, 3), 12.0);
    assert_eq!(res.at(3, 3), 8.0);
    assert_eq!(res.at(2, 2), 0f64);
}

#[test]
fn matmul_sparse() {
    let mut indexes: SparseMatrixData<f64> = HashMap::new();

    indexes.insert((0, 1), 2.0);
    indexes.insert((1, 0), 4.0);
    indexes.insert((1, 1), 6.0);
    indexes.insert((2, 2), 8.0);

    let sparse = SparseMatrix::<f64>::new(indexes, (3, 3));

    let mut indexes2: SparseMatrixData<f64> = HashMap::new();

    indexes2.insert((0, 0), 2.0);
    indexes2.insert((1, 0), 4.0);
    indexes2.insert((1, 1), 8.0);
    indexes2.insert((2, 1), 6.0);

    let sparse2 = SparseMatrix::<f64>::new(indexes2, (3, 3));

    let res = sparse.matmul_sparse(&sparse2).unwrap();

    assert_eq!(res.at(0, 0), 0.0);
    assert_eq!(res.at(0, 1), 24.0);
    assert_eq!(res.at(0, 2), 0.0);

    assert_eq!(res.at(1, 0), 8.0);
    assert_eq!(res.at(1, 1), 72.0);
    assert_eq!(res.at(1, 2), 0.0);

    assert_eq!(res.at(2, 0), 0.0);
    assert_eq!(res.at(2, 1), 0.0);
    assert_eq!(res.at(2, 2), 48.0);
}
