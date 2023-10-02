use sukker::Matrix;

fn main() {
    let a = Matrix::randomize_range(1f32, 100f32, (3, 1024));
    let b = Matrix::randomize_range(5f32, 100f32, (3, 1024));

    let any: bool = a.any(|&e| e >= 50f32);
    let all: bool = b.all(|&e| e >= 25f32);
    let sw: f32 = a.sum_where(|&e| e <= 33f32);
    let cw: usize = b.count_where(|&e| e >= 42f32);
    let f: Option<(usize, usize)> = a.find(|&e| e >= 50f32);
    let fa: Option<Vec<(usize, usize)>> = b.find_all(|&e| e >= 50f32);

    println!("Any returns: {}", any);
    println!("All returns: {}", all);
    println!("Sum where  returns: {}", sw);
    println!("Count where returns: {}", cw);
    println!("Find returns: {:?}", f);
    println!("Find all returns: {:?}", fa);
}
