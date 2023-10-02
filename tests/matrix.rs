use sukker::Matrix;

#[test]
fn basic() {
    let a = Matrix::init(3f32, (2, 3));
    let b = Matrix::init(5f32, (2, 3));

    let mut c = a.sub(&b).unwrap();
    assert_eq!(c.size(), 6);

    c.add_val_self(2.32);

    let c = c;

    let a: Matrix<f32> = Matrix::randomize((3, 2));

    let x = c.mm(&a).unwrap();

    assert_eq!(x.shape(), (2, 2));

    assert!(x.any(|&e| e >= 0.0));
}

#[test]
fn new() {
    let a = Matrix::<f32>::randomize((7, 56));
    let b = Matrix::<f32>::randomize((56, 8));

    let c = a.mm(&b).unwrap();

    // To print this beautiful matrix:
    c.print(7);
}
