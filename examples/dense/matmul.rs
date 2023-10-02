use sukker::Matrix;

fn main() {
    let a: Matrix<i32> = Matrix::randomize((3, 1024));
    let b: Matrix<i32> = Matrix::randomize((1024, 3));

    let c = a.matmul(&b).unwrap();

    c.print(2);

    assert!(c.size() == 9);
}
