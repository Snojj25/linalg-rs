use sukker::SparseMatrix;

fn main() {
    let matrix = SparseMatrix::<i32>::eye(100);

    println!("Sparsity: {}", matrix.sparsity());
}
