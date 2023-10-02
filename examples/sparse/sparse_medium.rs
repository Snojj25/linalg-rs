use std::collections::HashMap;
use sukker::{smd, SparseMatrix, SparseMatrixData};

fn main() {
    let mut indexes: SparseMatrixData<f64> = HashMap::new();

    indexes.insert((0, 1), 2.0);
    indexes.insert((1, 0), 4.0);
    indexes.insert((2, 3), 6.0);
    indexes.insert((3, 3), 8.0);

    let sparse = SparseMatrix::<f64>::new(indexes, (4, 4));

    // Or you can insert values like this
    let mut indexes2: SparseMatrixData<f64> =
        smd![((0, 0), 2.0), ((1, 0), 4.0), ((1, 1), 8.0), ((2, 3), 6.0)];

    let sparse2 = SparseMatrix::<f64>::new(indexes2, (4, 4));

    let res = sparse.add(&sparse2).unwrap();

    assert_eq!(res.at(0, 0), 2.0);
    assert_eq!(res.at(0, 1), 2.0);
    assert_eq!(res.at(1, 0), 8.0);
    assert_eq!(res.at(1, 1), 8.0);
    assert_eq!(res.at(2, 3), 12.0);
    assert_eq!(res.at(3, 3), 8.0);
    assert_eq!(res.at(2, 2), 0f64);

    res.print(3);
}
