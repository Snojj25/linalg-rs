// macros.rs

// Calculates 1D index from row and col
#[macro_export]
macro_rules! at {
    ($row:expr, $col:expr, $ncols:expr) => {
        $row * $ncols + $col
    };
}

// Generates a hash map for the sparse matrix
// by adding and e
#[macro_export]
macro_rules! smd {
    ( $( ($k:expr, $v:expr) ),* ) => {{
        let mut map = HashMap::new();
        $(
            map.insert($k, $v);
        )*
        map
    }};
}

pub(crate) use at;
pub(crate) use smd;
