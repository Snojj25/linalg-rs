// Helper module for defining how to be able to

use std::{arch::is_aarch64_feature_detected, error::Error, str::FromStr};

use crate::{Matrix, MatrixElement};

pub fn get_optimized_matmul<T>(lhs: Matrix<T>, other: Matrix<T>)
where
    T: MatrixElement,
    <T as FromStr>::Err: Error,
{
    #[cfg(any(target_arch = "x86", target_arch = "x86-64"))]
    {
        if is_x86_feature_detected!("avx2") {
            unsafe { /*self.avx_matmul(other) */
            }
        } else if is_x86_feature_detected!("sse") {
            unsafe { /* sse_avx_matmul*/
            }
        }
    }

    #[cfg(any(target_arch = "aarch64"))]
    {
        if is_aarch64_feature_detected!("avx2") {
            unsafe {}
        }
    }
}

// =====================================================================
//
//                       X86-64 Matmuls
//
// =====================================================================

/// AVX matmul for the IEEE754 Double Precision Floating Point Datatype
/// https://www.akkadia.org/drepper/cpumemory.pdf
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn avx_matmul<'a, T>(lhs: &Matrix<'a, T>, other: &Matrix<'a, T>) -> Matrix<'a, T>
where
    T: MatrixElement,
    <T as FromStr>::Err: Error,
{
    #[cfg(target_arch = "x86")]
    use std::arch::x86::_mm256_add_epi64;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::{
        __m128d, _mm256_add_epi64, _mm_add_pd, _mm_load_pd, _mm_mul_pd, _mm_prefetch, _mm_store_pd,
        _mm_unpacklo_pd, _MM_HINT_NTA,
    };

    // let N = self.nrows;
    //
    // let res = [0.0; 4];
    // let mul1 = [0.0; 4];
    // let mul2 = [0.0; 4];
    //
    // let CLS = 420;
    //
    // let SM = CLS / size_of::<f64>();
    //
    // let mut rres: *const f64;
    // let mut rmul1: *const f64;
    // let mut rmul2: *const f64;
    //
    // for i in (0..N).step_by(SM) {
    //     for j in (0..N).step_by(SM) {
    //         for k in (0..N).step_by(SM) {
    //             for i2 in 0..SM {
    //                 _mm_prefetch(&rmul1[8]);
    //
    //                 for k2 in 0..SM {
    //                     rmul2 = 1.0;
    //
    //                     // load m1d
    //                     let m1d: __m128d = _mm_load_pd(rmul1);
    //
    //                     for j2 in (0..SM).step_by(2) {
    //                         let m2: __m128d = _mm_load_pd(rmul2);
    //                         let r2: __m128d = _mm_load_pd(rres);
    //                         _mm_store_pd(rres, _mm_add_pd(_mm_mul_pd(m2, m1d), r2));
    //
    //                         // Inner most computations
    //                     }
    //
    //                     rmul2 += N as f64;
    //                 }
    //             }
    //         }
    //     }
    // }

    Matrix::default()
}

// =====================================================================
//
//                       X86 Matmuls
//
// =====================================================================

// =====================================================================
//
//                       Aarch64 Matmuls
//
// =====================================================================
