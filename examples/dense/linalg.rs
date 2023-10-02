use sukker::Matrix;

fn main() {
    let a: Matrix<f32> = Matrix::randomize((3, 1024));
    let b: Matrix<f32> = Matrix::randomize((1024, 3));

    let c = a.matmul(&b).unwrap();

    // Or alternatively
    let c = a.mm(&b).unwrap();

    println!("{}", c);
    println!("{:?}", c);

    // Print with however many decimals you want
    c.print(4);

    assert!(c.size() == 9);
}
