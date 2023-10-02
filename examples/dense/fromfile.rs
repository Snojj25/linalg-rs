use sukker::Matrix;

fn main() {
    let path = "../file.txt";

    let a: Matrix<f32> = Matrix::from_file(path).unwrap_or(Matrix::<f32>::default());

    a.print(3);
}
