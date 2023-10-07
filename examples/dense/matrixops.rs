use linalg_rs::constants::EF64;
use linalg_rs::Matrix;

fn main() {
    let a = Matrix::<f64>::randomize((2, 3));
    let b = Matrix::<f64>::randomize((2, 3));

    let mut c = a.add(&b).unwrap();

    // Will not multiply if dimensions does not line up
    c.mul_self(&b);

    let d = c.add_val(42f64).pow(3.0).dot(&b).unwrap().sub_val(EF64);

    d.print(5);
}
